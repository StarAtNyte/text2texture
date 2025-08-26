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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.1.0", "torchvision>=0.16.0", "diffusers>=0.30.0", "transformers",
        "accelerate", "peft", "safetensors", "Pillow", "fastapi",
        "uvicorn", "python-multipart", "numpy<2.0", "tqdm", "sentencepiece"
    ])
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0", "git"])
    .add_local_dir(".", "/app")  # Add current directory to container
)

app = modal.App("texture-generator-sdxl-part1", image=image)

# Model configuration for SDXL
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
DTYPE = torch.bfloat16

# Volumes
model_volume = modal.Volume.from_name("texture-model-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("generated-textures-sdxl", create_if_missing=True)

# SDXL-adapted seamless generation functions
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
            
            # Load refiner
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
        # Apply seamless techniques at 80% mark (original implementation)
        if step_index == int(pipe.num_timesteps * 0.8):
            make_seamless_sdxl(pipe.unet)
            make_seamless_sdxl(pipe.vae)

        # Noise Rolling: For the first 80% of steps, shift noise with 64px (original implementation)
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
    ) -> Image.Image:
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
                    generator=generator,
                    callback_on_step_end=self.sdxl_diffusion_callback
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


def load_materials_from_json(json_filename="materials_part1.json"):
    """Load materials from specified JSON file."""
    json_path = os.path.join("/app", json_filename)  # Use /app path from add_local_dir
    try:
        with open(json_path, 'r') as f:
            materials = json.load(f)
            print(f"Loaded materials from {json_path}")
            return materials
    except FileNotFoundError:
        print(f"Warning: {json_path} not found. Using fallback materials.")
        # Fallback to embedded data if file not found
        materials = {
            "rugs": {
                "hand-knotted-silk": {
                    "prompt_template": "seamless tileable hand-knotted silk rug material texture, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases, vector-clean edges, 4x upscale ready, crisp details, plain solid material texture only, no decorative patterns, no motifs, no designs, uniform surface",
                    "styles": ["fine-silk-knot-bumps", "dense-knot-surface", "lustrous-silk-weave"],
                    "colors": ["ivory", "champagne"]
                },
                "hand-knotted-wool": {
                    "prompt_template": "seamless tileable hand-knotted wool rug material texture, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases, vector-clean edges, 4x upscale ready, crisp details, plain solid material texture only, no decorative patterns, no motifs, no designs, uniform surface",
                    "styles": ["wool-knot-bumps", "thick-pile-surface", "matte-wool-texture"],
                    "colors": ["burgundy", "charcoal-gray"]
                }
            }
        }
        return materials
    except json.JSONDecodeError as e:
        print(f"Error parsing {json_path}: {e}")
        return {}


def enhance_prompt_for_details(prompt: str, seamless: bool = True) -> str:
    """
    Enhanced prompt engineering for flat material textures, optimized for SDXL,
    4x upscaling, and staying within the 77-token limit.

    This version prioritizes concise, high-impact technical terms.
    """
    # Core concepts are broken down into small, powerful groups.
    # We select the best, non-redundant terms.

    # 1. Seamlessness: The most direct and effective terms.
    seamless_keywords = "seamless, tileable" if seamless else ""

    # 2. Flatness & Lighting: Technical terms from 3D graphics that force a non-photographic, flat look.
    # 'orthographic' is much stronger than 'top-down'. 'diffuse map' or 'albedo' is key.
    material_keywords = "orthographic, texture map, diffuse map, flat lighting ,flat material texture, surface material"

    # 3. Detail & Quality: A mix of resolution, focus, and micro-detail keywords.
    # 'high frequency details' is a powerful term for generating fine grain.
    detail_keywords = "ultra detailed, 8k, sharp focus, macro photography, high frequency details, 4x upscale ready, perfect at zoom, maintains clarity when magnified"

    # Combine the components into a list to avoid messy string formatting.
    all_keywords = [
        prompt,  # The original base prompt comes first.
        seamless_keywords,
        material_keywords,
        detail_keywords,
    ]

    # Join them with commas, filtering out any empty strings (like when seamless=False).
    # This creates a clean, comma-separated string.
    enhanced_prompt = ", ".join(filter(None, all_keywords))

    return enhanced_prompt
    


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": output_volume, "/models": model_volume},
    timeout=86400,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def generate_texture_dataset():
    """Generate texture dataset from materials JSON."""
    
    # Configuration
    config = {
        "output_folder": "/data/generated_textures_part1",
        "guidance_scales": [7.5, 8.5],  # Higher values for sharper results
        "image_size": (1024, 1024),
        "num_inference_steps": 50,
        "seeds_per_material": 2,  # Number of different seeds per material variation
        "use_refiner": True,
        "high_noise_frac": 0.8,
        "negative_prompt": "blurry, soft focus, blur, soft, fuzzy, out of focus, unsharp, low resolution, low quality, poor quality, bad quality, lowres, pixelated, jpeg artifacts, compression artifacts, grainy, noisy, watermark, text, signature, logo, folds, creases, perspective, 3d effects, dimensional, shadows, lighting effects, depth, curved surface, bent material, folded, crumpled, seams, visible edges, discontinuity, mismatched patterns, asymmetric, uneven, irregular spacing, broken pattern, cartoon, illustration, deformed, artifacts, painting, drawing, clothing, fabric draping, person, object, scene, frame, border, people, hands, ugly, distorted, out of frame, poorly drawn, photorealistic, realistic photograph, photography, studio lighting, product shot, smooth, soft edges, lack of detail, plain, simple, cinematic lighting, dramatic lighting, volumetric lighting, ambient occlusion, raytracing, realistic materials, pbr materials, 3d render, cgi, rendered, pixelation when zoomed, loss of detail on magnification, upscaling artifacts, zoom degradation, scale-dependent quality loss, resolution limitations, clarity loss at high zoom, detail degradation, magnification blur, upscale quality loss, decorative patterns, ornamental designs, geometric patterns, floral patterns, motifs, ornaments, decorations, embellishments, designs, detailed patterns, complex patterns, intricate designs, ornate details, decorative elements, pattern designs, artistic patterns, stylized patterns, decorative motifs, ornamental elements, patterned designs, figured patterns, designed elements, ornamental patterns, decorative textures with patterns, patterned surfaces, designed textures, ornamental textures"
    }
    
    # Create output directories
    os.makedirs(config["output_folder"], exist_ok=True)
    
    # Load materials
    materials = load_materials_from_json()
    
    # Initialize generator
    generator = SDXLTextureGenerator()
    
    # Track generation statistics
    total_images = 0
    successful_generations = 0
    failed_generations = 0
    generated_image_paths = []  # Track all successfully generated images
    
    # Calculate total target images
    total_combinations = 0
    for category_name, category_data in materials.items():
        for material_name, material_data in category_data.items():
            styles = len(material_data["styles"])
            colors = len(material_data["colors"])
            total_combinations += styles * colors
    
    total_target_images = total_combinations * len(config["guidance_scales"]) * config["seeds_per_material"]
    
    print(f"Starting texture generation")
    print(f"Total material combinations: {total_combinations}")
    print(f"Guidance scales: {config['guidance_scales']}")
    print(f"Seeds per combination: {config['seeds_per_material']}")
    print(f"Total target images: {total_target_images}")
    
    # Process each category
    for category_name, category_data in materials.items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category_name.upper()}")
        print(f"Materials in category: {len(category_data)}")
        print(f"{'='*60}")
        
        # Create category directory
        category_dir = Path(config["output_folder"]) / category_name
        os.makedirs(category_dir, exist_ok=True)
        
        # Process each material
        for material_name, material_data in category_data.items():
            print(f"\nProcessing material: {material_name}")
            print(f"Styles: {len(material_data['styles'])}, Colors: {len(material_data['colors'])}")
            
            # Generate textures for each style-color combination
            for style in material_data["styles"]:
                for color in material_data["colors"]:
                    # Create prompt from template
                    original_prompt = material_data["prompt_template"].format(style=style, color=color)
                    enhanced_prompt = enhance_prompt_for_details(original_prompt, seamless=True)
                    
                    print(f"  Style: {style}, Color: {color}")
                    
                    # Generate with different guidance scales and seeds
                    for guidance_scale in config["guidance_scales"]:
                        for seed_idx in range(config["seeds_per_material"]):
                            try:
                                # Generate seed based on combination for consistency
                                seed_string = f"{category_name}_{material_name}_{style}_{color}_{guidance_scale}_{seed_idx}"
                                seed = hash(seed_string) % (2**32)
                                if seed < 0:
                                    seed += 2**32
                                
                                print(f"    Generating: G={guidance_scale}, Seed={seed_idx+1}/{config['seeds_per_material']}")
                                
                                # Generate texture image
                                image, actual_seed = generator.generate_texture_image(
                                    prompt=enhanced_prompt,
                                    negative_prompt=config["negative_prompt"],
                                    width=config["image_size"][0],
                                    height=config["image_size"][1],
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=config["num_inference_steps"],
                                    seed=seed,
                                    enable_seamless=True,
                                    use_refiner=config["use_refiner"],
                                    high_noise_frac=config["high_noise_frac"]
                                )
                                
                                # Create filename
                                sanitized_material = _sanitize_filename(material_name)
                                sanitized_style = _sanitize_filename(style)
                                sanitized_color = _sanitize_filename(color)
                                
                                filename = f"tex_{sanitized_material}_{sanitized_style}_{sanitized_color}_g{guidance_scale}_seed{actual_seed}_sdxl_seamless.png"
                                image_path = category_dir / filename
                                
                                # Save image
                                image.save(image_path, "PNG", quality=95)
                                
                                print(f"      ✓ Saved: {filename}")
                                
                                successful_generations += 1
                                total_images += 1
                                generated_image_paths.append(str(image_path))  # Track generated image path
                                
                                # Clear GPU memory
                                torch.cuda.empty_cache()
                                
                            except Exception as e:
                                print(f"      ✗ Failed: {str(e)}")
                                failed_generations += 1
                                total_images += 1
                                continue
    
    # Final statistics
    print(f"\n{'='*60}")
    print("TEXTURE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Overall success rate: {(successful_generations/total_images*100):.1f}%")
    print(f"Guidance scales used: {config['guidance_scales']}")
    print(f"Seeds per combination: {config['seeds_per_material']}")
    print(f"Output directory: {config['output_folder']}")
    
    # Create summary file
    summary = {
        "generation_config": config,
        "statistics": {
            "total_images_processed": total_images,
            "successful_generations": successful_generations,
            "failed_generations": failed_generations,
            "success_rate": successful_generations/total_images*100 if total_images > 0 else 0,
            "total_combinations": total_combinations,
            "categories_processed": list(materials.keys()),
            "seeds_per_combination": config["seeds_per_material"],
            "guidance_scales": config["guidance_scales"]
        },
        "model_info": {
            "model_id": MODEL_ID,
            "refiner_id": REFINER_ID,
            "guidance_scales": config["guidance_scales"],
            "num_inference_steps": config["num_inference_steps"],
            "image_size": config["image_size"],
            "use_refiner": config["use_refiner"]
        }
    }
    
    summary_path = Path(config["output_folder"]) / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return generated_image_paths


@app.function(
    image=image,
    volumes={"/data": output_volume},
    timeout=3600  # 1 hour
)
def collect_generated_images():
    """Phase 2: Collect all generated images from Phase 1 for further processing."""
    import glob
    
    output_folder = "/data/generated_textures"
    
    if not os.path.exists(output_folder):
        print(f"Output folder {output_folder} not found. Run Phase 1 first.")
        return []
    
    # Find all generated PNG files
    pattern = os.path.join(output_folder, "**", "*.png")
    generated_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(generated_files)} generated images in {output_folder}")
    
    # Filter for SDXL seamless files
    sdxl_files = [f for f in generated_files if "sdxl_seamless.png" in f]
    print(f"Found {len(sdxl_files)} SDXL seamless texture files")
    
    return sdxl_files


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": output_volume, "/models": model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def test_seamless_settings():
    """Test different seamless threshold settings and create tiled comparisons."""
    import numpy as np
    
    # Test configuration - use real enhanced prompt
    base_prompt = "seamless tileable hand-knotted wool rug material texture, wool-knot-bumps, burgundy"
    test_prompt = enhance_prompt_for_details(base_prompt, seamless=True)
    negative_prompt = "blurry, soft focus, blur, soft, fuzzy, out of focus, unsharp, low resolution, low quality, poor quality, bad quality, lowres, pixelated, jpeg artifacts, compression artifacts, grainy, noisy, watermark, text, signature, logo, folds, creases, perspective, 3d effects, dimensional, shadows, lighting effects, depth, curved surface, bent material, folded, crumpled, seams, visible edges, discontinuity, mismatched patterns, asymmetric, uneven, irregular spacing, broken pattern, cartoon, illustration, deformed, artifacts, painting, drawing, clothing, fabric draping, person, object, scene, frame, border, people, hands, ugly, distorted, out of frame, poorly drawn, photorealistic, realistic photograph, photography, studio lighting, product shot, smooth, soft edges, lack of detail, plain, simple, cinematic lighting, dramatic lighting, volumetric lighting, ambient occlusion, raytracing, realistic materials, pbr materials, 3d render, cgi, rendered, pixelation when zoomed, loss of detail on magnification, upscaling artifacts, zoom degradation, scale-dependent quality loss, resolution limitations, clarity loss at high zoom, detail degradation, magnification blur, upscale quality loss, decorative patterns, ornamental designs, geometric patterns, floral patterns, motifs, ornaments, decorations, embellishments, designs, detailed patterns, complex patterns, intricate designs, ornate details, decorative elements, pattern designs, artistic patterns, stylized patterns, decorative motifs, ornamental elements, patterned designs, figured patterns, designed elements, ornamental patterns, decorative textures with patterns, patterned surfaces, designed textures, ornamental textures"
    
    output_dir = "/data/seamless_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Different threshold values and rolling shifts to test
    test_configs = [
        {'threshold': 0.3, 'shift': 0},    # No rolling
        {'threshold': 0.5, 'shift': 0},    # No rolling
        {'threshold': 0.7, 'shift': 0},    # No rolling
        {'threshold': 0.9, 'shift': 0},    # No rolling
        {'threshold': 0.3, 'shift': 1},    # Minimal rolling
        {'threshold': 0.5, 'shift': 1},    # Minimal rolling
        {'threshold': 0.7, 'shift': 1},    # Minimal rolling
        {'threshold': 0.9, 'shift': 1},    # Minimal rolling
    ]
    
    generator = SDXLTextureGenerator()
    results = []
    
    for config in test_configs:
        threshold = config['threshold']
        shift_amount = config['shift']
        print(f"Testing threshold: {threshold}, shift: {shift_amount}")
        
        # Temporarily modify the callback threshold
        original_callback = generator.sdxl_diffusion_callback
        
        def test_callback(pipe, step_index, timestep, callback_kwargs):
            # Apply seamless at specified threshold (like original repo)
            if step_index == int(pipe.num_timesteps * threshold):
                make_seamless_sdxl(pipe.unet)
                make_seamless_sdxl(pipe.vae)
            
            # Variable noise rolling based on test config
            if step_index < int(pipe.num_timesteps * threshold) and shift_amount > 0:
                callback_kwargs["latents"] = torch.roll(
                    callback_kwargs["latents"], 
                    shifts=(shift_amount, shift_amount), 
                    dims=(2, 3)
                )
            
            return callback_kwargs
        
        generator.sdxl_diffusion_callback = test_callback
        
        try:
            # Generate test image at full resolution
            image, seed = generator.generate_texture_image(
                prompt=test_prompt,
                negative_prompt=negative_prompt,
                width=1024,  # Full resolution
                height=1024,
                guidance_scale=7.5,
                num_inference_steps=50,  # Full steps like real generation
                seed=42,  # Fixed seed for comparison
                enable_seamless=True,
                use_refiner=True  # Use refiner like real generation
            )
            
            # Create 2x2 tiled version to check seamlessness
            tiled = Image.new('RGB', (2048, 2048))
            tiled.paste(image, (0, 0))
            tiled.paste(image, (1024, 0))
            tiled.paste(image, (0, 1024))
            tiled.paste(image, (1024, 1024))
            
            # Save both single and tiled versions
            single_path = f"{output_dir}/test_th{threshold}_shift{shift_amount}_single.png"
            tiled_path = f"{output_dir}/test_th{threshold}_shift{shift_amount}_tiled.png"
            
            image.save(single_path)
            tiled.save(tiled_path)
            
            results.append({
                'threshold': threshold,
                'shift': shift_amount,
                'single_path': single_path,
                'tiled_path': tiled_path,
                'seed': seed
            })
            
            print(f"✓ Generated test for threshold {threshold}, shift {shift_amount}")
            
        except Exception as e:
            print(f"✗ Failed threshold {threshold}: {e}")
        
        # Restore original callback
        generator.sdxl_diffusion_callback = original_callback
        torch.cuda.empty_cache()
    
    # Create summary image combining all tiled results (4x2 grid)
    if results:
        summary = Image.new('RGB', (4096, 2048))
        positions = [(i*512, j*1024) for j in range(2) for i in range(8)]
        
        for i, result in enumerate(results[:8]):
            if i < len(positions):
                try:
                    tiled_img = Image.open(result['tiled_path'])
                    # Resize to fit in grid
                    tiled_img = tiled_img.resize((512, 512))
                    summary.paste(tiled_img, positions[i])
                except Exception as e:
                    print(f"Error adding result {i} to summary: {e}")
        
        summary_path = f"{output_dir}/seamless_comparison_summary.png"
        summary.save(summary_path)
        print(f"Summary comparison saved: {summary_path}")
    
    return results

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": output_volume, "/models": model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def test_multiple_materials():
    """Test multiple materials with optimal settings: threshold 0.9, shift 1."""
    
    output_dir = "/data/material_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load materials from JSON
    materials = load_materials_from_json()
    
    # Original implementation settings
    threshold = 0.8
    shift_amount = 64
    
    generator = SDXLTextureGenerator()
    results = []
    
    # Custom callback with optimal settings
    def optimal_callback(pipe, step_index, timestep, callback_kwargs):
        # Apply seamless at 90% mark
        if step_index == int(pipe.num_timesteps * threshold):
            make_seamless_sdxl(pipe.unet)
            make_seamless_sdxl(pipe.vae)
        
        # Minimal noise rolling before seamless
        if step_index < int(pipe.num_timesteps * threshold):
            callback_kwargs["latents"] = torch.roll(
                callback_kwargs["latents"], 
                shifts=(shift_amount, shift_amount), 
                dims=(2, 3)
            )
        
        return callback_kwargs
    
    # Replace callback
    generator.sdxl_diffusion_callback = optimal_callback
    
    # Test different material types
    test_materials = []
    for category_name, category_data in materials.items():
        for material_name, material_data in category_data.items():
            # Take first style and color for each material
            if material_data["styles"] and material_data["colors"]:
                test_materials.append({
                    'category': category_name,
                    'material': material_name,
                    'style': material_data["styles"][0],
                    'color': material_data["colors"][0],
                    'template': material_data["prompt_template"]
                })
    
    print(f"Testing {len(test_materials)} materials with ORIGINAL settings: threshold={threshold}, shift={shift_amount}")
    
    for i, mat in enumerate(test_materials[:6]):  # Test first 6 materials
        try:
            # Create prompt
            base_prompt = mat['template'].format(style=mat['style'], color=mat['color'])
            enhanced_prompt = enhance_prompt_for_details(base_prompt, seamless=True)
            
            print(f"[{i+1}/{len(test_materials[:6])}] Generating {mat['category']}/{mat['material']}")
            
            # Generate texture
            image, seed = generator.generate_texture_image(
                prompt=enhanced_prompt,
                negative_prompt="blurry, soft focus, blur, patterns, motifs, designs, decorative elements, ornamental, artifacts",
                width=1024,
                height=1024,
                guidance_scale=7.5,
                num_inference_steps=50,
                seed=42 + i,  # Different seed for each material
                enable_seamless=True,
                use_refiner=True
            )
            
            # Create 2x2 tiled version
            tiled = Image.new('RGB', (2048, 2048))
            tiled.paste(image, (0, 0))
            tiled.paste(image, (1024, 0))
            tiled.paste(image, (0, 1024))
            tiled.paste(image, (1024, 1024))
            
            # Save files with original settings marker
            safe_name = f"{_sanitize_filename(mat['category'])}_{_sanitize_filename(mat['material'])}_original"
            single_path = f"{output_dir}/{safe_name}_single.png"
            tiled_path = f"{output_dir}/{safe_name}_tiled.png"
            
            image.save(single_path)
            tiled.save(tiled_path)
            
            results.append({
                'material': f"{mat['category']}/{mat['material']}",
                'single_path': single_path,
                'tiled_path': tiled_path,
                'seed': seed
            })
            
            print(f"✓ Saved {safe_name}")
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Failed {mat['category']}/{mat['material']}: {e}")
            continue
    
    # Create summary grid
    if results:
        grid_size = min(3, len(results))  # 3x2 or smaller grid
        summary = Image.new('RGB', (3 * 1024, 2 * 1024))
        positions = [(i * 1024, j * 1024) for j in range(2) for i in range(3)]
        
        for i, result in enumerate(results[:6]):
            if i < len(positions):
                try:
                    single_img = Image.open(result['single_path'])
                    summary.paste(single_img, positions[i])
                except Exception as e:
                    print(f"Error adding result {i} to summary: {e}")
        
        summary_path = f"{output_dir}/materials_summary_original.png"
        summary.save(summary_path)
        print(f"Summary saved: {summary_path}")
    
    return results

@app.local_entrypoint()
def test():
    """Run seamless threshold tests."""
    results = test_seamless_settings.remote()
    print(f"Test completed with {len(results)} threshold variations")
    return results

@app.local_entrypoint()
def test_materials():
    """Run material tests with optimal settings."""
    results = test_multiple_materials.remote()
    print(f"Material test completed with {len(results)} textures")
    return results

@app.local_entrypoint()
def main():
    """Local entry point to run Phase 1 only - texture generation."""
    generate_texture_dataset.remote()


@app.local_entrypoint() 
def phase2():
    """Local entry point to run Phase 2 independently."""
    image_paths = collect_generated_images.remote()
    print(f"Phase 2 collected {len(image_paths)} images for processing")
    return image_paths


@app.function(
    image=image,
    volumes={"/data": output_volume},
    timeout=3600  # 1 hour
)
def tile_textures_4x4():
    """Phase 3: Create 4x4 tiled versions of all generated textures."""
    from PIL import Image
    
    # Configuration
    config = {
        "input_folder": "/data/generated_textures_part1",
        "output_folder": "/data/tiled_textures_part1",
        "batch_size": 50  # Can process many more since it's just PIL operations
    }
    
    # Create output directory
    os.makedirs(config["output_folder"], exist_ok=True)
    
    print(f"Starting 4x4 tiling process...")
    print(f"Input folder: {config['input_folder']}")
    print(f"Output folder: {config['output_folder']}")
    
    # Check if input folder exists and has images
    if not os.path.exists(config["input_folder"]):
        print(f"Error: Input folder {config['input_folder']} does not exist.")
        return []
    
    # Find all generated PNG files
    import glob
    pattern = os.path.join(config["input_folder"], "**", "*.png")
    generated_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(generated_files)} images to tile.")
    
    if len(generated_files) == 0:
        print("No images found to tile.")
        return []
    
    # Process images in batches
    total_batches = (len(generated_files) + config["batch_size"] - 1) // config["batch_size"]
    successful_count = 0
    failed_count = 0
    tiled_paths = []
    
    for i in range(0, len(generated_files), config["batch_size"]):
        batch_num = (i // config["batch_size"]) + 1
        batch = generated_files[i:i + config["batch_size"]]
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} images)...")
        
        for j, img_path in enumerate(batch):
            try:
                # Load the image
                with Image.open(img_path) as img:
                    # Get original size (should be 1024x1024)
                    width, height = img.size
                    
                    # Create 4x4 tiled image (4096x4096)
                    tiled = Image.new('RGB', (width * 4, height * 4))
                    
                    # Paste the image 16 times in a 4x4 grid
                    for row in range(4):
                        for col in range(4):
                            x = col * width
                            y = row * height
                            tiled.paste(img, (x, y))
                    
                    # Create output filename
                    base_name, ext = os.path.splitext(os.path.basename(img_path))
                    if not base_name.endswith('_tiled4x4'):
                        base_name = base_name + '_tiled4x4'
                    output_name = f"{base_name}{ext}"
                    output_path = os.path.join(config["output_folder"], output_name)
                    
                    # Handle filename conflicts
                    counter = 1
                    while os.path.exists(output_path):
                        output_path = os.path.join(config["output_folder"], f"{base_name}_{counter}{ext}")
                        counter += 1
                    
                    # Save the tiled image
                    tiled.save(output_path, "PNG", optimize=True)
                    tiled_paths.append(output_path)
                    
                    successful_count += 1
                    
                    # Progress within batch
                    if (j + 1) % 10 == 0 or (j + 1) == len(batch):
                        print(f"  Progress: {j + 1}/{len(batch)} images in batch {batch_num}")
                
            except Exception as e:
                print(f"  ✗ Failed to tile {os.path.basename(img_path)}: {e}")
                failed_count += 1
                continue
        
        print(f"  ✓ Batch {batch_num} complete: {len(batch)} images processed")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("4X4 TILING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {len(generated_files)}")
    print(f"Successful tiles: {successful_count}")
    print(f"Failed tiles: {failed_count}")
    print(f"Success rate: {(successful_count/len(generated_files)*100):.1f}%")
    print(f"Output directory: {config['output_folder']}")
    print(f"Output size: 4096x4096 (4x4 grid of 1024x1024 tiles)")
    
    # Create summary file
    summary = {
        "tiling_config": config,
        "statistics": {
            "total_images_input": len(generated_files),
            "successful_tiles": successful_count,
            "failed_tiles": failed_count,
            "success_rate": successful_count/len(generated_files)*100 if generated_files else 0,
            "output_paths": tiled_paths[:10]  # First 10 paths as sample
        },
        "tiling_info": {
            "method": "4x4_seamless_tile",
            "input_size": "1024x1024",
            "output_size": "4096x4096",
            "tiles_per_image": 16,
            "grid_size": "4x4"
        }
    }
    
    summary_path = Path(config["output_folder"]) / "tiling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return tiled_paths


@app.local_entrypoint()
def tile():
    """Local entry point to create 4x4 tiled versions of all generated textures."""
    print("Starting 4x4 tiling process...")
    tiled_paths = tile_textures_4x4.remote()
    print(f"Tiling complete. Processed {len(tiled_paths)} images.")
    return tiled_paths

@app.local_entrypoint()
def full_pipeline():
    """Local entry point to run both Phase 1 and Phase 2."""
    print("Starting Phase 1: Texture Generation...")
    generated_paths = generate_texture_dataset.remote()
    
    print(f"\nPhase 1 complete. Generated {len(generated_paths)} images.")
    print("Starting Phase 2: Image Collection...")
    
    all_image_paths = collect_generated_images.remote()
    print(f"Phase 2 complete. Found {len(all_image_paths)} total images for processing.")
    
    return all_image_paths

@app.local_entrypoint()
def complete_pipeline():
    """Local entry point to run texture generation and tiling."""
    print("Starting Complete Pipeline: Generation + Tiling...")
    
    print("Phase 1: Texture Generation...")
    generated_paths = generate_texture_dataset.remote()
    print(f"Phase 1 complete. Generated {len(generated_paths)} images.")
    
    print("\nPhase 2: Image Collection...")
    all_image_paths = collect_generated_images.remote()
    print(f"Phase 2 complete. Found {len(all_image_paths)} total images.")
    
    print("\nPhase 3: 4x4 Tiling...")
    tiled_paths = tile_textures_4x4.remote()
    print(f"Phase 3 complete. Tiled {len(tiled_paths)} images.")
    
    print("\nComplete pipeline finished!")
    return tiled_paths


if __name__ == "__main__":
    # Local execution - runs full pipeline by default
    with app.run():
        print("Starting Phase 1: Texture Generation...")
        generated_paths = generate_texture_dataset.remote()
        
        print(f"\nPhase 1 complete. Generated {len(generated_paths)} images.")
        print("Starting Phase 2: Image Collection...")
        
        all_image_paths = collect_generated_images.remote()
        print(f"Phase 2 complete. Found {len(all_image_paths)} total images for processing.")