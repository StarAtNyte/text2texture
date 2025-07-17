import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os
import sys
import subprocess
import shutil
import glob
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import re
import random
import inspect

# --- Configuration ---
load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")

if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file. Model download might fail if it's gated.")

# --- Path Configurations ---
try:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_ROOT = os.getcwd()

REPOS_DIR = os.path.join(APP_ROOT, "repositories")
MODELS_STORAGE_DIR = os.path.join(APP_ROOT, "pretrained_models_storage")
SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_DIR = os.path.join(REPOS_DIR, "SwinIR")
MODEL_URLS = {
    "SwinIR_L_x4_GAN.pth": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "path": os.path.join(MODELS_STORAGE_DIR, "swinir", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    }
}
TEMP_INPUT_DIR_BASE = os.path.join(APP_ROOT, "temp_inputs_sr")
TEMP_OUTPUT_DIR_BASE = os.path.join(APP_ROOT, "temp_outputs_sr")
SR_MODEL_TYPE = "SwinIR-L"
SR_PATCH_WISE = False
SR_TILE_SIZE = "640"

# --- Helper Functions ---
def run_command(command_list, cwd=None, desc=None):
    if desc:
        print(desc)
    env = os.environ.copy()
    if cwd:
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{cwd}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = cwd
    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True, env=env)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {' '.join(command_list)}")
        print(f"Return code: {process.returncode}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
        raise RuntimeError(f"Command failed. Check console for details. Stderr: {stderr or stdout}")
    return stdout, stderr

def _sanitize_filename(text: str, max_length: int = 100) -> str:
    sanitized = re.sub(r'[^\w\s-]', '', text)
    sanitized = re.sub(r'[\s/]+', '_', sanitized).strip('-_')
    return sanitized[:max_length]

# --- Environment and Model Setup ---
def setup_environment():
    print("Performing environment setup for SwinIR...")
    os.makedirs(REPOS_DIR, exist_ok=True)
    os.makedirs(MODELS_STORAGE_DIR, exist_ok=True)
    os.makedirs(TEMP_INPUT_DIR_BASE, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR_BASE, exist_ok=True)
    if not os.path.exists(SWINIR_DIR):
        print(f"Cloning SwinIR into {SWINIR_DIR}...")
        run_command(["git", "clone", SWINIR_REPO_URL, SWINIR_DIR])
        print("SwinIR cloned. Ensure its dependencies are installed.")
    else:
        print(f"SwinIR repository found at {SWINIR_DIR}.")
    for model_name, model_info in MODEL_URLS.items():
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                run_command(["wget", model_info["url"], "-O", model_path])
            except (FileNotFoundError, RuntimeError):
                print("wget failed, trying curl...")
                run_command(["curl", "-L", model_info["url"], "-o", model_path, "--create-dirs"])
        else:
            print(f"SwinIR model {model_name} found at {model_path}.")
    print("SwinIR setup check complete.")

# --- Expanded Texture Categories ---
TEXTURE_CATEGORIES = {
    "rugs": {
        "wool-pile": {
            "prompt_template": "colormap, seamless tileable wool-pile rug, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["cut-pile", "loop-pile", "saxony", "frieze"],
            "colors": ["charcoal", "indigo", "sage", "terracotta"]
        },
        "jute-boucle": {
            "prompt_template": "colormap, seamless tileable jute-boucle rug, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["chunky", "fine", "mixed-yarn", "herringbone"],
            "colors": ["natural", "walnut", "moss", "stone"]
        },
        "sisal-flatweave": {
            "prompt_template": "colormap, seamless tileable sisal-flatweave rug, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["basket", "herringbone", "chevron", "plain"],
            "colors": ["sand", "khaki", "charcoal", "indigo"]
        },
        "persian-traditional": {
            "prompt_template": "colormap, seamless tileable persian traditional rug, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["tabriz", "kashan", "isfahan", "heriz"],
            "colors": ["burgundy", "navy", "ivory", "gold"]
        }
    },
    "fabrics": {
        "cotton-canvas": {
            "prompt_template": "colormap, seamless tileable cotton canvas fabric, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["plain-weave", "twill", "duck-canvas", "heavy-weight"],
            "colors": ["natural", "navy", "khaki", "black"]
        },
        "linen-textile": {
            "prompt_template": "colormap, seamless tileable linen textile fabric, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["fine-weave", "loose-weave", "slub-texture", "stonewashed"],
            "colors": ["flax", "white", "gray", "sage"]
        },
        "denim-fabric": {
            "prompt_template": "colormap, seamless tileable denim fabric, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["raw-selvage", "washed", "stretch", "heavyweight"],
            "colors": ["indigo", "black", "light-wash", "dark-wash"]
        },
        "silk-textile": {
            "prompt_template": "colormap, seamless tileable silk textile fabric, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["charmeuse", "taffeta", "dupioni", "chiffon"],
            "colors": ["ivory", "champagne", "burgundy", "navy"]
        }
    },
    "papers": {
        "watercolor-paper": {
            "prompt_template": "colormap, seamless tileable watercolor paper, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["hot-press", "cold-press", "rough", "medium"],
            "colors": ["white", "cream", "natural", "ivory"]
        },
        "kraft-paper": {
            "prompt_template": "colormap, seamless tileable kraft paper, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["smooth", "textured", "recycled", "heavy-weight"],
            "colors": ["brown", "natural", "bleached", "dark-brown"]
        },
        "handmade-paper": {
            "prompt_template": "colormap, seamless tileable handmade paper, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["mulberry", "bamboo", "cotton-rag", "hemp"],
            "colors": ["natural", "white", "cream", "gray"]
        },
        "parchment-paper": {
            "prompt_template": "colormap, seamless tileable parchment paper, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["aged", "smooth", "textured", "antique"],
            "colors": ["cream", "beige", "aged-yellow", "light-brown"]
        }
    },
    "leather": {
        "full-grain-leather": {
            "prompt_template": "colormap, seamless tileable full-grain leather, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["smooth", "pebbled", "pull-up", "distressed"],
            "colors": ["brown", "black", "cognac", "burgundy"]
        },
        "suede-leather": {
            "prompt_template": "colormap, seamless tileable suede leather, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["nubuck", "brushed", "soft-suede", "microsuede"],
            "colors": ["tan", "gray", "navy", "black"]
        }
    },
    "wood": {
        "hardwood-grain": {
            "prompt_template": "colormap, seamless tileable hardwood grain, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["oak", "maple", "walnut", "cherry"],
            "colors": ["natural", "honey", "dark-stain", "ebony"]
        },
        "reclaimed-wood": {
            "prompt_template": "colormap, seamless tileable reclaimed wood, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["barn-wood", "driftwood", "weathered", "rustic"],
            "colors": ["gray", "brown", "weathered-gray", "natural"]
        }
    },
    "metal": {
        "brushed-metal": {
            "prompt_template": "colormap, seamless tileable brushed metal, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["aluminum", "steel", "brass", "copper"],
            "colors": ["silver", "gold", "bronze", "gunmetal"]
        },
        "oxidized-metal": {
            "prompt_template": "colormap, seamless tileable oxidized metal, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["patina", "rust", "verdigris", "aged"],
            "colors": ["green-patina", "rust-orange", "brown-rust", "blue-patina"]
        }
    }
}

# Configuration for reduced color/cfg combinations
CFG_VALUES = [6.5, 7.5]  # Reduced from 5 to 2 values

# SDXL + LoRA Configuration
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_ID = "dog-god/texture-synthesis-sdxl-lora"
LORA_WEIGHT_NAME = "texture-synthesis-topdown-base-condensed.safetensors"
LORA_SCALE = 0.8

def generate_texture_plan():
    """Generates a comprehensive texture plan from the expanded categories."""
    plan = []
    
    for category_name, category_data in TEXTURE_CATEGORIES.items():
        for material_name, material_data in category_data.items():
            for style in material_data["styles"]:
                for color in material_data["colors"]:
                    for cfg in CFG_VALUES:
                        plan.append({
                            "category": category_name,
                            "material": material_name,
                            "style": style,
                            "color": color,
                            "prompt": material_data["prompt_template"].format(style=style, color=color),
                            "guidance_scale": cfg
                        })
    
    print(f"Generated texture plan with {len(plan)} total combinations")
    print(f"Categories: {len(TEXTURE_CATEGORIES)}")
    print(f"Total materials: {sum(len(cat) for cat in TEXTURE_CATEGORIES.values())}")
    print(f"Colors per material: 4, CFG values: {len(CFG_VALUES)}")
    return plan

# --- Core Classes and Functions ---
class TextToTextureGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        if not torch.cuda.is_available() and device == "cuda":
            print("CUDA not available, falling back to CPU. This will be very slow.")
            self.device = "cpu"

    def initialize_sd_pipeline(self, model_id=MODEL_ID):
        if self.pipe is not None:
            print("Stable Diffusion XL pipeline already initialized.")
            return
        print(f"Initializing Stable Diffusion XL Pipeline: {model_id} on {self.device}...")
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            # Use DPM++ scheduler for better quality
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Load texture synthesis LoRA
            print(f"Loading texture synthesis LoRA: {LORA_ID}")
            self.pipe.load_lora_weights(LORA_ID, weight_name=LORA_WEIGHT_NAME)
            
            self.pipe.to(self.device)
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            print("Stable Diffusion XL Pipeline with LoRA initialized successfully.")
        except Exception as e:
            print(f"Error initializing Stable Diffusion XL pipeline: {e}")
            raise

    def offload_sd_model(self):
        if hasattr(self, 'pipe') and self.pipe is not None:
            print("Offloading Stable Diffusion model to CPU and deleting...")
            try:
                self.pipe.to('cpu')
            except Exception as e:
                print(f"Note: Error moving pipeline to CPU: {e}")
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
        print("Stable Diffusion model offloaded.")

    def generate_single_variation(self, prompt: str, output_dir: str, filename_prefix: str, guidance_scale: float, seed: int,
                                  negative_prompt: str, width: int, height: int, num_inference_steps: int) -> Path:
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized. Call initialize_sd_pipeline() first.")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        base_name = _sanitize_filename(filename_prefix)
        print(f"  SD Gen: G={guidance_scale}, Seed={seed} -> '{base_name}'")
        
        generator = torch.Generator(device="cpu" if self.device == "cpu" else self.device).manual_seed(seed)
        
        try:
            with torch.inference_mode():
                image = self.pipe(
                    prompt=prompt, negative_prompt=negative_prompt,
                    width=width, height=height,
                    num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                    generator=generator,
                    cross_attention_kwargs={"scale": LORA_SCALE}
                ).images[0]

            output_filename = f"{base_name}_g{guidance_scale}_seed{seed}_sdxl_orig.png"
            image_path = Path(output_dir) / output_filename
            
            counter = 0
            while image_path.exists():
                counter += 1
                output_filename = f"{base_name}_g{guidance_scale}_seed{seed}_sdxl_orig_{counter}.png"
                image_path = Path(output_dir) / output_filename
            
            image.save(image_path)
            return image_path

        except Exception as e:
            print(f"  SD Gen: ERROR for seed {seed} with prefix '{base_name}': {e}")
            if "out of memory" in str(e).lower() and self.device == "cuda":
                print("  CUDA OOM. Try reducing image dimensions (width/height).")
            return None

def enhance_prompt_for_details(prompt: str) -> str:
    detail_keywords = "sharp focus, 4k texture quality, intricate details"
    if detail_keywords.lower() not in prompt.lower():
        return f"{prompt}, {detail_keywords}"
    return prompt

def file_already_exists(output_dir: str, filename_prefix: str, guidance_scale: float, seed: int, suffix: str = "_sdxl_orig.png") -> bool:
    """Check if a file with the given parameters already exists."""
    base_name = _sanitize_filename(filename_prefix)
    expected_filename = f"{base_name}_g{guidance_scale}_seed{seed}{suffix}"
    expected_path = Path(output_dir) / expected_filename
    return expected_path.exists()

def sr_file_already_exists(output_dir: str, original_filename: str) -> bool:
    """Check if super-resolved version already exists."""
    # Remove the original extension and add the SR suffix
    base_name = Path(original_filename).stem
    sr_filename = f"{base_name}_swinir_x4.png"
    sr_path = Path(output_dir) / sr_filename
    return sr_path.exists()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        setup_environment()
    except Exception as e:
        print(f"CRITICAL: Environment setup failed: {e}. Processing might fail.")
        sys.exit(1)

    # --- Generation Phase ---
    generation_plan = generate_texture_plan()
    if not generation_plan:
        print("No generation tasks found in plan. Exiting.")
        sys.exit(0)

    output_directory_sd = "generated_textures_from_plan_sd"
    Path(output_directory_sd).mkdir(parents=True, exist_ok=True)

    sd_generator = TextToTextureGenerator()
    
    print("\n--- PHASE 1: Stable Diffusion XL + LoRA Texture Generation ---")
    sd_generator.initialize_sd_pipeline()
    
    # Define the default negative prompt optimized for SDXL
    default_negative_prompt = "blurry, lowres, watermark, text, signature, folds, creases, perspective, 3d, cartoon, illustration, deformed, artifacts, jpeg artifacts, painting, drawing, clothing, fabric draping, shadows, depth, curved surface, bent material, folded, crumpled, seams, stitches, person, object, scene, frame, border, logo, people, hands, ugly, distorted, out of frame, poorly drawn"

    all_sd_image_paths = []
    total_images_to_generate = len(generation_plan)
    print(f"Total images to generate: {total_images_to_generate}")
    
    img_count = 0
    failed_generations = 0
    skipped_generations = 0
    
    for task in generation_plan:
        material = f"{task['category']}_{task['material']}"
        style = task['style']
        color = task['color']
        guidance = task['guidance_scale']
        original_prompt = task['prompt']
        
        img_count += 1
        print(f"\nImage {img_count}/{total_images_to_generate}: {material}_{style}_{color} (Guidance {guidance})")
        
        # Generate a consistent seed based on the texture parameters
        seed_string = f"{material}_{style}_{color}_{guidance}"
        seed = hash(seed_string) % (2**32)
        if seed < 0:
            seed += 2**32
        
        filename_prefix = f"tex_{_sanitize_filename(material)}_{_sanitize_filename(style)}_{_sanitize_filename(color)}"
        
        # Check if file already exists
        if file_already_exists(output_directory_sd, filename_prefix, guidance, seed):
            print(f"  SKIPPED: File already exists")
            skipped_generations += 1
            # Still add to paths list for super-resolution phase
            base_name = _sanitize_filename(filename_prefix)
            expected_filename = f"{base_name}_g{guidance}_seed{seed}_sdxl_orig.png"
            existing_path = Path(output_directory_sd) / expected_filename
            all_sd_image_paths.append(existing_path)
            continue
        
        enhanced_prompt = enhance_prompt_for_details(original_prompt)

        try:
            generated_path = sd_generator.generate_single_variation(
                prompt=enhanced_prompt,
                output_dir=output_directory_sd,
                filename_prefix=filename_prefix,
                guidance_scale=guidance,
                seed=seed,
                negative_prompt=default_negative_prompt,
                width=1024,
                height=1024, # Using square textures for consistency
                num_inference_steps=28
            )
            
            if generated_path:
                all_sd_image_paths.append(generated_path)
                print(f"  SUCCESS: Generated {generated_path.name}")
            else:
                failed_generations += 1
                print(f"  FAILED: Could not generate image")
                
        except Exception as e:
            failed_generations += 1
            print(f"  ERROR: Exception during generation: {e}")
            if "out of memory" in str(e).lower():
                print("  WARNING: CUDA OOM detected. Consider reducing batch size or image dimensions.")

    print(f"\n--- PHASE 1 COMPLETE: Stable Diffusion XL + LoRA textures generated. ---")
    print(f"Successfully generated: {len(all_sd_image_paths) - skipped_generations} images")
    print(f"Skipped existing: {skipped_generations} images")
    print(f"Failed generations: {failed_generations} images")
    print(f"Total images available for SR: {len(all_sd_image_paths)} images")
    
    sd_generator.offload_sd_model()
    print("Waiting a few seconds for VRAM to clear...")
    time.sleep(5)

    # --- Super-Resolution Phase ---
    print("\n--- PHASE 2: Super-Resolution using SwinIR ---")
    output_directory_sr = "generated_textures_from_plan_sr"
    Path(output_directory_sr).mkdir(parents=True, exist_ok=True)
    
    if all_sd_image_paths:
        print(f"Using superresolve.py to batch process {len(all_sd_image_paths)} images...")
        try:
            # Call the superresolve.py script as a subprocess
            superresolve_script = os.path.join(APP_ROOT, "superresolve.py")
            cmd = [
                sys.executable, superresolve_script,
                str(output_directory_sd),  # input folder
                str(output_directory_sr),  # output folder
                "--batch_size", "3",  # Smaller batch size for safety
                "--patch_wise",       # Enable patch-wise processing for lower VRAM
                "--delay_between_batches", "3.0"
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=APP_ROOT)
            
            if result.returncode == 0:
                print("Super-resolution completed successfully!")
                if result.stdout:
                    print("STDOUT:", result.stdout[-1000:])  # Show last 1000 chars to avoid too much output
            else:
                print(f"Super-resolution failed with return code {result.returncode}")
                if result.stderr:
                    print("STDERR:", result.stderr[-1000:])
                if result.stdout:
                    print("STDOUT:", result.stdout[-1000:])
        
        except Exception as e:
            print(f"Error running super-resolution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No images to super-resolve.")

    print("--- All Processing Complete ---")
    print(f"Original SD images are in: {output_directory_sd}")
    print(f"Super-resolved images are in: {output_directory_sr}")
    print(f"\nSummary:")
    print(f"- Total images planned: {total_images_to_generate}")
    print(f"- Successfully generated: {len(all_sd_image_paths)}")
    print(f"- Failed generations: {failed_generations}")
    print(f"- Super-resolution attempted on: {len(all_sd_image_paths)} images")