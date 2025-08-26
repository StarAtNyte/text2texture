import modal
import json
import base64
import io
import os
from pathlib import Path
from PIL import Image
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional
import random
from typing import List, Dict
import time
import re
from flask import Flask, render_template_string, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile

# Modal setup - exactly matching the original
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0", "torchvision>=0.16.0", "diffusers>=0.30.0", "transformers",
        "accelerate", "peft", "safetensors", "Pillow", "flask",
        "numpy<2.0", "tqdm", "sentencepiece", "werkzeug"
    ])
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "git"])
    .add_local_dir(".", "/app")
)

app_modal = modal.App("texture-webapp-full", image=image)

# Model configuration - exactly matching original
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
DTYPE = torch.bfloat16

# Volumes
model_volume = modal.Volume.from_name("texture-model-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("webapp-textures-full", create_if_missing=True)

# SDXL-adapted seamless generation functions - EXACTLY from original
def asymmetricConv2DConvForward_circular(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """Circular padding for Conv2d layers - adapted for SDXL architecture"""
    self.paddingX = (
        self._reversed_padding_repeated_twice[0],
        self._reversed_padding_repeated_twice[1],
        0,
        0
    )

    self.paddingY = (
        0,
        0,
        self._reversed_padding_repeated_twice[2],
        self._reversed_padding_repeated_twice[3]
    )
    working = F.pad(input, self.paddingX, mode="circular")
    working = F.pad(working, self.paddingY, mode="circular")

    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


def make_seamless_sdxl(model):
    """Enable circular padding on all Conv2d layers in SDXL model"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module._conv_forward = asymmetricConv2DConvForward_circular.__get__(module, Conv2d)


def disable_seamless_sdxl(model):
    """Disable circular padding and restore default Conv2d behavior"""
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            module._conv_forward = nn.Conv2d._conv_forward.__get__(module, Conv2d)


def _sanitize_filename(text: str, max_length: int = 100) -> str:
    sanitized = re.sub(r'[^\w\s-]', '', text)
    sanitized = re.sub(r'[\s/]+', '_', sanitized).strip('-_')
    return sanitized[:max_length]


def enhance_prompt_for_details(prompt: str, seamless: bool = True) -> str:
    """
    Enhanced prompt engineering for flat material textures, optimized for SDXL,
    4x upscaling, and staying within the 77-token limit.
    """
    seamless_keywords = "seamless, tileable" if seamless else ""
    material_keywords = "orthographic, texture map, diffuse map, flat lighting, flat material texture, surface material"
    detail_keywords = "ultra detailed, 8k, sharp focus, macro photography, high frequency details, 4x upscale ready, perfect at zoom, maintains clarity when magnified"

    all_keywords = [
        prompt,
        seamless_keywords,
        material_keywords,
        detail_keywords,
    ]

    enhanced_prompt = ", ".join(filter(None, all_keywords))
    return enhanced_prompt


class SDXLTextureGenerator:
    def __init__(self):
        self.model_id = MODEL_ID
        self.refiner_id = REFINER_ID
        self.pipe = None
        self.refiner = None

    def initialize_pipeline(self):
        """Initialize SDXL pipeline with optional refiner."""
        if self.pipe is None:
            from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
            
            print("Loading SDXL base model...")
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=DTYPE,
                use_safetensors=True
            )
            
            print("Loading SDXL refiner...")
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_id,
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=DTYPE,
                use_safetensors=True
            )
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            self.refiner.enable_attention_slicing()
            self.refiner.enable_vae_slicing()
            
            # Move to GPU
            self.pipe.to("cuda")
            self.refiner.to("cuda")
            print("SDXL models loaded successfully!")

    def sdxl_diffusion_callback(self, pipe, step_index, timestep, callback_kwargs):
        """
        Callback for seamless pattern generation adapted for SDXL architecture.
        Uses original pattern-diffusion implementation settings.
        """
        if step_index == int(pipe.num_timesteps * 0.8):
            make_seamless_sdxl(pipe.unet)
            make_seamless_sdxl(pipe.vae)

        if step_index < int(pipe.num_timesteps * 0.8):
            callback_kwargs["latents"] = torch.roll(
                callback_kwargs["latents"], 
                shifts=(64, 64), 
                dims=(2, 3)
            )

        return callback_kwargs

    def generate_texture_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = None,
        enable_seamless: bool = True,
        use_refiner: bool = True,
        high_noise_frac: float = 0.8,
        oversample_factor: float = 1
    ) -> tuple:
        """Generate a single texture image from prompt."""
        self.initialize_pipeline()
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Calculate oversized dimensions for cropping
        gen_width = int(width * oversample_factor)
        gen_height = int(height * oversample_factor)
        
        print(f"Generating {'seamless ' if enable_seamless else ''}texture with seed {seed}")
        print(f"Generation size: {gen_width}x{gen_height}, final size: {width}x{height}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Ensure seamless is disabled before starting (SDXL adaptation)
        if enable_seamless:
            if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                disable_seamless_sdxl(self.pipe.unet)
            if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                disable_seamless_sdxl(self.pipe.vae)
        
        # Generate without autocast to avoid precision issues
        torch.cuda.empty_cache()
        
        if enable_seamless:
            # Use SDXL seamless generation with callback
            if use_refiner:
                # Two-stage generation with refiner
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=num_inference_steps,
                    denoising_end=high_noise_frac,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=self.sdxl_diffusion_callback,
                    output_type="latent"
                ).images
                
                # Refine the image with seamless callback (keep oversized)
                image = self.refiner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    image=image,
                    generator=generator,
                    callback_on_step_end=self.sdxl_diffusion_callback
                ).images[0]
            else:
                # Single-stage generation
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=self.sdxl_diffusion_callback
                ).images[0]
        else:
            # Standard SDXL generation without seamless techniques
            if use_refiner:
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=num_inference_steps,
                    denoising_end=high_noise_frac,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent"
                ).images
                
                image = self.refiner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    image=image,
                    generator=generator
                ).images[0]
            else:
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=gen_width,
                    height=gen_height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
        
        # Crop to desired dimensions
        if oversample_factor > 1.0:
            # Calculate crop box for center region
            left = (gen_width - width) // 2
            top = (gen_height - height) // 2
            right = left + width
            bottom = top + height
            image = image.crop((left, top, right, bottom))
        
        return image, seed


# Professional UI with advanced features
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Text2Texture Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 2rem;
            align-items: start;
        }
        
        .control-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            position: sticky;
            top: 120px;
        }
        
        .results-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            min-height: 600px;
        }
        
        .form-section {
            margin-bottom: 2rem;
        }
        
        .form-section h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #4a5568;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .form-section h3:before {
            content: "";
            width: 4px;
            height: 20px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 2px;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 500;
            color: #2d3748;
            font-size: 0.9rem;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            background: #fff;
        }
        
        input:focus, textarea:focus, select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            height: 100px;
            resize: vertical;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        input[type="checkbox"] {
            width: auto;
            margin: 0;
        }
        
        .generate-btn {
            width: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 1rem;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 1rem;
        }
        
        .generate-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results-grid {
            display: grid;
            gap: 2rem;
        }
        
        .result-item {
            background: #f8fafc;
            border-radius: 16px;
            padding: 1.5rem;
            border: 2px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .result-item:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
        
        .result-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .result-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2d3748;
        }
        
        .result-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .image-container {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            background: #fff;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .image-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .image-label {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            color: white;
            padding: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .download-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .download-btn {
            flex: 1;
            padding: 0.5rem 1rem;
            background: #48bb78;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            text-align: center;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #38a169;
            transform: translateY(-1px);
        }
        
        .generation-info {
            background: #edf2f7;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 0.85rem;
            color: #4a5568;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        
        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 1rem;
            border-radius: 12px;
            margin: 1rem 0;
            border: 1px solid #feb2b2;
        }
        
        .preset-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .preset-btn {
            padding: 0.5rem;
            background: #edf2f7;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }
        
        .preset-btn:hover {
            background: #e2e8f0;
            border-color: #cbd5e0;
        }
        
        .preset-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        @media (max-width: 1200px) {
            .container {
                grid-template-columns: 1fr;
                max-width: 800px;
            }
            
            .control-panel {
                position: static;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>🎨 Professional Text2Texture Generator</h1>
    </nav>
    
    <div class="container">
        <div class="control-panel">
            <form id="textureForm">
                <div class="form-section">
                    <h3>📝 Texture Prompt</h3>
                    <div class="form-group">
                        <label for="prompt">Describe your texture:</label>
                        <textarea id="prompt" name="prompt" placeholder="seamless wood grain texture, dark oak, weathered surface, high detail" required></textarea>
                    </div>
                    
                    <div class="preset-buttons">
                        <div class="preset-btn" onclick="setPreset('wood')">Wood Texture</div>
                        <div class="preset-btn" onclick="setPreset('fabric')">Fabric Material</div>
                        <div class="preset-btn" onclick="setPreset('stone')">Stone Surface</div>
                        <div class="preset-btn" onclick="setPreset('metal')">Metal Texture</div>
                    </div>
                </div>
                
                <div class="form-section">
                    <h3>⚙️ Generation Settings</h3>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="width">Width:</label>
                            <select id="width" name="width">
                                <option value="512">512px</option>
                                <option value="1024" selected>1024px</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="height">Height:</label>
                            <select id="height" name="height">
                                <option value="512">512px</option>
                                <option value="1024" selected>1024px</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="guidance_scale">Guidance Scale:</label>
                            <input type="number" id="guidance_scale" name="guidance_scale" min="1" max="20" step="0.5" value="7.5">
                        </div>
                        <div class="form-group">
                            <label for="steps">Inference Steps:</label>
                            <input type="number" id="steps" name="steps" min="20" max="100" value="50">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="seed">Seed (optional):</label>
                            <input type="number" id="seed" name="seed" placeholder="Random if empty">
                        </div>
                        <div class="form-group">
                            <label for="oversample">Oversample Factor:</label>
                            <select id="oversample" name="oversample">
                                <option value="1" selected>1x (No Oversampling)</option>
                                <option value="1.2">1.2x</option>
                                <option value="1.5">1.5x</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="form-section">
                    <h3>🔧 Advanced Options</h3>
                    <div class="checkbox-group">
                        <input type="checkbox" id="seamless" name="seamless" checked>
                        <label for="seamless">Enable Seamless Mode</label>
                    </div>
                    
                    <div class="checkbox-group">
                        <input type="checkbox" id="use_refiner" name="use_refiner" checked>
                        <label for="use_refiner">Use SDXL Refiner</label>
                    </div>
                    
                    <div class="form-group">
                        <label for="high_noise_frac">High Noise Fraction:</label>
                        <input type="number" id="high_noise_frac" name="high_noise_frac" min="0.1" max="1.0" step="0.05" value="0.8">
                    </div>
                    
                    <div class="form-group">
                        <label for="tile_size">Tiling Options:</label>
                        <select id="tile_size" name="tile_size">
                            <option value="2">2x2 Tile Test</option>
                            <option value="3">3x3 Tile Test</option>
                            <option value="4">4x4 Tile Test</option>
                        </select>
                    </div>
                </div>
                
                <button type="submit" class="generate-btn">
                    🚀 Generate Professional Texture
                </button>
            </form>
        </div>
        
        <div class="results-panel">
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <h3>Generating High-Quality Texture...</h3>
                <p>This process uses SDXL with advanced seamless techniques.<br>
                Expected time: 2-3 minutes for optimal quality.</p>
            </div>
            
            <div id="results">
                <div style="text-align: center; color: #718096; padding: 3rem;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">🎨</div>
                    <h2>Professional Texture Generator</h2>
                    <p>Create high-quality, seamless textures using SDXL technology.<br>
                    Configure your settings and click generate to begin.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const presets = {
            wood: "seamless wood grain texture, dark oak, weathered surface, natural knots, high detail macro",
            fabric: "seamless fabric material texture, cotton weave, neutral beige, textile surface, detailed fiber",
            stone: "seamless stone surface texture, granite, rough natural texture, mineral details, weathered rock", 
            metal: "seamless metal surface texture, brushed steel, industrial finish, metallic reflection, oxidation details"
        };
        
        function setPreset(type) {
            document.getElementById('prompt').value = presets[type];
            
            // Update active preset button
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        document.getElementById('textureForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loadingDiv = document.getElementById('loading');
            const resultsDiv = document.getElementById('results');
            const submitButton = this.querySelector('button[type="submit"]');
            
            // Show loading, hide previous results
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = '';
            submitButton.disabled = true;
            submitButton.innerHTML = '⏳ Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const tileSize = formData.get('tile_size');
                    resultsDiv.innerHTML = `
                        <div class="result-item">
                            <div class="result-header">
                                <div class="result-title">✨ Generated Professional Texture</div>
                            </div>
                            
                            <div class="result-images">
                                <div class="image-container">
                                    <img src="/image/${result.image_id}" alt="Generated Texture" />
                                    <div class="image-label">Original Texture (${result.width}x${result.height})</div>
                                </div>
                                <div class="image-container">
                                    <img src="/image/${result.tiled_id}" alt="Tiled Texture" />
                                    <div class="image-label">${tileSize}x${tileSize} Seamless Test</div>
                                </div>
                            </div>
                            
                            <div class="download-actions">
                                <a href="/download/${result.image_id}" class="download-btn">
                                    📥 Download Original
                                </a>
                                <a href="/download/${result.tiled_id}" class="download-btn">
                                    📥 Download Tiled
                                </a>
                            </div>
                            
                            <div class="generation-info">
                                <strong>Generation Details:</strong>
                                <div class="info-grid">
                                    <div><strong>Seed:</strong> ${result.seed}</div>
                                    <div><strong>Size:</strong> ${result.width}x${result.height}</div>
                                    <div><strong>Seamless:</strong> ${result.seamless ? 'Enabled' : 'Disabled'}</div>
                                    <div><strong>Refiner:</strong> ${result.use_refiner ? 'Used' : 'Skipped'}</div>
                                    <div><strong>Steps:</strong> ${result.steps}</div>
                                    <div><strong>Guidance:</strong> ${result.guidance_scale}</div>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    resultsDiv.innerHTML = `<div class="error">❌ Generation Error: ${result.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error">❌ Network Error: ${error.message}</div>`;
            } finally {
                loadingDiv.style.display = 'none';
                submitButton.disabled = false;
                submitButton.innerHTML = '🚀 Generate Professional Texture';
            }
        });
    </script>
</body>
</html>
"""

# Flask app with full SDXL integration
flask_app = Flask(__name__)
texture_generator = None
stored_images = {}

@flask_app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@flask_app.route('/generate', methods=['POST'])
def generate_texture():
    global texture_generator, stored_images
    
    try:
        # Initialize generator if needed
        if texture_generator is None:
            texture_generator = SDXLTextureGenerator()
        
        # Get form data
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'})
        
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 1024))
        guidance_scale = float(request.form.get('guidance_scale', 7.5))
        steps = int(request.form.get('steps', 50))
        seamless = request.form.get('seamless') == 'on'
        use_refiner = request.form.get('use_refiner') == 'on'
        high_noise_frac = float(request.form.get('high_noise_frac', 0.8))
        oversample = float(request.form.get('oversample', 1.0))
        tile_size = int(request.form.get('tile_size', 2))
        
        seed = request.form.get('seed')
        if seed:
            seed = int(seed)
        else:
            seed = None
        
        # Enhance prompt using the exact same function
        enhanced_prompt = enhance_prompt_for_details(prompt, seamless=seamless)
        
        # Use the EXACT same negative prompt as the original
        negative_prompt = ("blurry, soft focus, blur, soft, fuzzy, out of focus, unsharp, "
                          "low resolution, low quality, poor quality, bad quality, lowres, "
                          "pixelated, jpeg artifacts, compression artifacts, grainy, noisy, "
                          "watermark, text, signature, logo, folds, creases, perspective, "
                          "3d effects, dimensional, shadows, lighting effects, depth, "
                          "curved surface, bent material, folded, crumpled, seams, visible edges, "
                          "discontinuity, mismatched patterns, asymmetric, uneven, irregular spacing, "
                          "broken pattern, cartoon, illustration, deformed, artifacts, painting, drawing, "
                          "clothing, fabric draping, person, object, scene, frame, border, people, hands, "
                          "ugly, distorted, out of frame, poorly drawn, photorealistic, realistic photograph, "
                          "photography, studio lighting, product shot, smooth, soft edges, lack of detail, "
                          "plain, simple, cinematic lighting, dramatic lighting, volumetric lighting, "
                          "ambient occlusion, raytracing, realistic materials, pbr materials, 3d render, "
                          "cgi, rendered, pixelation when zoomed, loss of detail on magnification, "
                          "upscaling artifacts, zoom degradation, scale-dependent quality loss, "
                          "resolution limitations, clarity loss at high zoom, detail degradation, "
                          "magnification blur, upscale quality loss, decorative patterns, ornamental designs, "
                          "geometric patterns, floral patterns, motifs, ornaments, decorations, embellishments, "
                          "designs, detailed patterns, complex patterns, intricate designs, ornate details, "
                          "decorative elements, pattern designs, artistic patterns, stylized patterns, "
                          "decorative motifs, ornamental elements, patterned designs, figured patterns, "
                          "designed elements, ornamental patterns, decorative textures with patterns, "
                          "patterned surfaces, designed textures, ornamental textures")
        
        # Generate texture using EXACT same parameters
        image, actual_seed = texture_generator.generate_texture_image(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            enable_seamless=seamless,
            use_refiner=use_refiner,
            high_noise_frac=high_noise_frac,
            oversample_factor=oversample
        )
        
        # Create tiled version based on user selection
        tiled_image = Image.new('RGB', (width * tile_size, height * tile_size))
        for row in range(tile_size):
            for col in range(tile_size):
                x = col * width
                y = row * height
                tiled_image.paste(image, (x, y))
        
        # Generate unique IDs for images
        image_id = f"texture_{actual_seed}_{int(time.time())}"
        tiled_id = f"tiled_{tile_size}x{tile_size}_{actual_seed}_{int(time.time())}"
        
        # Store images in memory
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        stored_images[image_id] = img_byte_arr.getvalue()
        
        tiled_byte_arr = io.BytesIO()
        tiled_image.save(tiled_byte_arr, format='PNG')
        stored_images[tiled_id] = tiled_byte_arr.getvalue()
        
        return jsonify({
            'success': True,
            'image_id': image_id,
            'tiled_id': tiled_id,
            'seed': actual_seed,
            'width': width,
            'height': height,
            'seamless': seamless,
            'use_refiner': use_refiner,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'tile_size': f"{tile_size}x{tile_size}"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@flask_app.route('/image/<image_id>')
def get_image(image_id):
    if image_id not in stored_images:
        return "Image not found", 404
    
    return send_file(
        io.BytesIO(stored_images[image_id]),
        mimetype='image/png',
        as_attachment=False
    )

@flask_app.route('/download/<image_id>')
def download_image(image_id):
    if image_id not in stored_images:
        return "Image not found", 404
    
    return send_file(
        io.BytesIO(stored_images[image_id]),
        mimetype='image/png',
        as_attachment=True,
        download_name=f"{image_id}.png"
    )

# Modal web endpoints - properly configured
@app_modal.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": model_volume},
    timeout=7200,  # 2 hours for complex generations
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.web_endpoint(method="GET", label="texture-webapp-full")
def web_app_get():
    return flask_app

@app_modal.function(
    image=image,
    gpu="A100-40GB", 
    volumes={"/models": model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
@modal.web_endpoint(method="POST", label="texture-webapp-full")  
def web_app_post():
    return flask_app

if __name__ == "__main__":
    # For local testing
    flask_app.run(debug=True, host="0.0.0.0", port=5000)