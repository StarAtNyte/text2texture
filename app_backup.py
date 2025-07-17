import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import os
import sys
import subprocess # For running SwinIR
import shutil # For file operations
import glob # For finding SwinIR output files
import time
import uuid # For unique temporary folder names
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import re
import random # For generating random seeds if none are provided
import inspect # For getting default negative prompt

# --- Configuration ---
# Load Hugging Face token from .env file
load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")

# Log in to Hugging Face Hub (required for some models like SD3)
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in .env file. Model download might fail if it's gated.")

# --- Path Configurations (Adapted from SR script) ---
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
SR_PATCH_WISE = False # Set to True if SwinIR still causes OOM with large SD outputs
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
    sanitized = re.sub(r'\s+', '_', sanitized).strip('-_')
    return sanitized[:max_length]

# --- Setup Function (Adapted from SR script) ---
def setup_environment():
    print("Performing environment setup for SwinIR...")
    os.makedirs(REPOS_DIR, exist_ok=True)
    os.makedirs(MODELS_STORAGE_DIR, exist_ok=True)
    os.makedirs(TEMP_INPUT_DIR_BASE, exist_ok=True)
    os.makedirs(TEMP_OUTPUT_DIR_BASE, exist_ok=True)

    if not os.path.exists(SWINIR_DIR):
        print(f"Cloning SwinIR into {SWINIR_DIR}...")
        run_command(["git", "clone", SWINIR_REPO_URL, SWINIR_DIR])
        print("SwinIR cloned. Ensure its dependencies (e.g., 'timm') are installed (`pip install timm`).")
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
                print("wget failed or not found, trying curl...")
                try:
                    run_command(["curl", "-L", model_info["url"], "-o", model_path, "--create-dirs"])
                except Exception as e:
                    print(f"CRITICAL: Failed to download {model_name} using both wget and curl. Error: {e}")
                    raise
        else:
            print(f"SwinIR model {model_name} found at {model_path}.")
    print("SwinIR setup check complete.")

# --- SwinIR Inference Function (Adapted from SR script) ---
def process_with_swinir(image_path_str: str, model_type: str = SR_MODEL_TYPE, test_patch_wise: bool = SR_PATCH_WISE, tile_size: str = SR_TILE_SIZE) -> str:
    print(f"SwinIR: Processing {os.path.basename(image_path_str)} with {model_type}, patch-wise: {test_patch_wise}")
    script_path = os.path.join(SWINIR_DIR, "main_test_swinir.py")
    unique_id = str(uuid.uuid4())

    if model_type == "SwinIR-L":
        model_file = MODEL_URLS["SwinIR_L_x4_GAN.pth"]["path"]
        output_subfolder_in_repo_results = "swinir_real_sr_x4_large"
    else:
        raise ValueError(f"Unsupported SwinIR model type: {model_type}. Expected SwinIR-L.")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"SwinIR model file not found: {model_file}. Run setup_environment or check paths.")

    temp_swinir_input_dir = os.path.join(TEMP_INPUT_DIR_BASE, unique_id, "swinir_lq")
    os.makedirs(temp_swinir_input_dir, exist_ok=True)
    temp_input_image_path = os.path.join(temp_swinir_input_dir, os.path.basename(image_path_str))
    shutil.copy(image_path_str, temp_input_image_path)

    swinir_repo_results_dir = os.path.join(SWINIR_DIR, "results")
    if os.path.exists(swinir_repo_results_dir):
        shutil.rmtree(swinir_repo_results_dir)

    cmd = [
        sys.executable, script_path,
        "--task", "real_sr", "--model_path", model_file,
        "--folder_lq", temp_swinir_input_dir, "--scale", "4"
    ]
    if model_type == "SwinIR-L": cmd.append("--large_model")
    if test_patch_wise: cmd.extend(["--tile", tile_size])

    run_command(cmd, cwd=SWINIR_DIR, desc=f"SwinIR: Running {model_type} on {os.path.basename(image_path_str)}...")

    original_basename = os.path.splitext(os.path.basename(image_path_str))[0]
    expected_output_filename_in_repo = f"{original_basename}_SwinIR.png"
    output_image_path_in_repo = os.path.join(swinir_repo_results_dir, output_subfolder_in_repo_results, expected_output_filename_in_repo)

    if not os.path.exists(output_image_path_in_repo):
        glob_pattern = os.path.join(swinir_repo_results_dir, output_subfolder_in_repo_results, f"{original_basename}*.png")
        possible_files = glob.glob(glob_pattern)
        if not possible_files:
            raise FileNotFoundError(f"SwinIR output file not found. Expected at '{output_image_path_in_repo}' or via glob '{glob_pattern}'.")
        output_image_path_in_repo = possible_files[0]
        print(f"SwinIR: Found output via glob: {output_image_path_in_repo}")

    controlled_temp_output_dir = os.path.join(TEMP_OUTPUT_DIR_BASE, unique_id) # This dir will contain just the SR image
    os.makedirs(controlled_temp_output_dir, exist_ok=True)
    # Use a generic name for the temp SR file, actual final naming happens in main loop
    final_temp_sr_path = os.path.join(controlled_temp_output_dir, f"sr_output_{unique_id}.png")
    shutil.move(output_image_path_in_repo, final_temp_sr_path)

    shutil.rmtree(temp_swinir_input_dir, ignore_errors=True)
    if os.path.exists(swinir_repo_results_dir):
        shutil.rmtree(swinir_repo_results_dir, ignore_errors=True)
    
    return final_temp_sr_path # This is the path to the SR image in its unique temp output folder

class TextToTextureGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda"):
        self.device = device
        self.pipe = None # Initialize pipe as None
        if not torch.cuda.is_available() and device == "cuda":
            print("CUDA not available, falling back to CPU for Stable Diffusion. This will be very slow.")
            self.device = "cpu"
        # Pipeline initialization moved to a separate method to be called explicitly
        
    def initialize_sd_pipeline(self, model_id="stabilityai/stable-diffusion-3.5-medium"):
        if self.pipe is not None:
            print("Stable Diffusion pipeline already initialized.")
            return

        print(f"Initializing Stable Diffusion 3 Pipeline with model: {model_id} on {self.device}...")
        try:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            self.pipe.to(self.device)
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            # Consider enable_model_cpu_offload if VRAM is extremely tight even for SD alone
            # if hasattr(self.pipe, 'enable_model_cpu_offload') and self.device == "cuda":
            #      print("Enabling model CPU offload for Stable Diffusion pipeline.")
            #      self.pipe.enable_model_cpu_offload()
            print("Stable Diffusion Pipeline initialized successfully.")
        except Exception as e:
            print(f"Error initializing Stable Diffusion 3 pipeline: {e}")
            raise

    def offload_sd_model(self):
        if hasattr(self, 'pipe') and self.pipe is not None:
            print("Offloading Stable Diffusion model to CPU and deleting...")
            try:
                self.pipe.to('cpu') # Move components to CPU first
            except Exception as e:
                print(f"Note: Error moving pipeline to CPU (might be normal if already partially offloaded): {e}")
            del self.pipe
            self.pipe = None
            print("Stable Diffusion pipeline object deleted.")
        
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            # Optional: Check VRAM after clearing
            # if self.device == "cuda":
            #     print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            #     print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("Stable Diffusion model offloaded and CUDA cache cleared.")

    def generate_two_variations(self,
                                prompt: str,
                                output_dir: str = "generated_textures",
                                filename_prefix: str = "texture",
                                negative_prompt: str = "blurry, low quality, unrealistic, noisy, text, watermark, signature, deformed, artifacts, jpeg artifacts, illustration, cartoon, painting, drawing, folds, wrinkles, creases, clothing, fabric draping, shadows, 3d depth, perspective, curved surface, bent material, folded, crumpled, seams, stitches, person, object, scene, frame, border, signature, logo, people, hands, ugly, distorted, out of frame, poorly drawn",
                                width: int = 1024, # Adjusted for potentially less VRAM during SD phase if SwinIR is very heavy later
                                height: int = 1376, # Adjusted to square, common for textures. Original: 1376
                                num_inference_steps: int = 28,
                                guidance_scale: float = 4.5,
                                seeds: list = None) -> list:
        if self.pipe is None: # Check if pipeline is initialized
            print("Pipeline not initialized. Call initialize_sd_pipeline() first. Cannot generate images.")
            return [] # Return empty list

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if seeds is None or not isinstance(seeds, list) or len(seeds) != 2:
            seeds_to_use = [
                torch.randint(0, 2**32 - 1, (1,)).item(),
                torch.randint(0, 2**32 - 1, (1,)).item()
            ]
            while seeds_to_use[0] == seeds_to_use[1]:
                seeds_to_use[1] = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seeds_to_use = seeds

        print(f"\nSD Gen: Prompt: \"{prompt}\"")
        print(f"SD Gen: Params: W={width}, H={height}, Steps={num_inference_steps}, Guidance={guidance_scale}, Seeds={seeds_to_use}")

        generated_sd_paths = []

        if filename_prefix == "prompt": base_name = _sanitize_filename(prompt)
        else: base_name = _sanitize_filename(filename_prefix)

        for i, seed_val in enumerate(seeds_to_use):
            print(f"  SD Gen: Variation {i+1} with seed: {seed_val}...")
            generator = torch.Generator(device="cpu" if self.device == "cpu" else self.device).manual_seed(seed_val)
            
            original_image_path = None
            try:
                with torch.inference_mode():
                    image = self.pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        width=width, height=height,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        generator=generator
                    ).images[0]

                output_filename_orig = f"{base_name}_seed{seed_val}_sd_orig.png"
                original_image_path = Path(output_dir) / output_filename_orig
                
                counter = 0
                while original_image_path.exists():
                    counter += 1
                    output_filename_orig = f"{base_name}_seed{seed_val}_sd_orig_{counter}.png"
                    original_image_path = Path(output_dir) / output_filename_orig
                
                image.save(original_image_path)
                print(f"  SD Gen: Original texture saved to: {original_image_path}")
                generated_sd_paths.append(original_image_path)

            except Exception as e:
                print(f"  SD Gen: Error for variation {i+1} (seed {seed_val}): {e}")
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    print("  SD Gen: CUDA OOM. Try reducing image dimensions (width/height) or enable model_cpu_offload.")
                # Do not append None here, only successful paths
        return generated_sd_paths

def enhance_prompt_for_details(prompt: str) -> str:
    detail_keywords = "sharp focus, 4k texture quality, intricate details"
    if detail_keywords.lower() not in prompt.lower():
        return f"{prompt}, {detail_keywords}"
    return prompt

if __name__ == "__main__":
    try:
        setup_environment() # Prepare SwinIR repo and models first
    except Exception as e:
        print(f"CRITICAL: SwinIR Environment setup failed: {e}. SwinIR processing might fail later.")

    output_directory = "generated_rug_textures_detailed_sr_sequential"
    Path(output_directory).mkdir(parents=True, exist_ok=True) # Ensure output dir exists early

    sd_generator = TextToTextureGenerator() # device defaults to "cuda" if available
    
    # --- Phase 1: Generate all SD images ---
    print("\n--- PHASE 1: Stable Diffusion Texture Generation ---")
    sd_generator.initialize_sd_pipeline() # Load SD model into VRAM
    
    all_sd_image_paths_to_super_resolve = []
    
    texture_prompts_configs = [
        {
            "prompt": "Flat wool pile texture, dense natural wool fibers, uniform cream color, material surface only, soft pile texture, flat view, seamless repeating pattern",
            "filename_prefix": "wool_pile_texture",
            "seed_pair": [12345, 54321]
        },
        {
            "prompt": "Flat wool loop pile texture, uniform gray wool loops, dense berber-style surface, material texture only, flat appearance, seamless texture",
            "filename_prefix": "wool_loop_texture",
        },
        {
            "prompt": "Flat merino wool texture, fine soft fibers, uniform ivory color, dense surface, material texture only, flat view, seamless tileable",
            "filename_prefix": "merino_wool_texture",
            "seed_pair": [55555, 55556]
        },
        {
            "prompt": "Flat lambswool texture, curly soft fibers, uniform pearl gray color, fluffy pile surface, material texture only, flat appearance, repeating pattern",
            "filename_prefix": "lambswool_texture",
            "seed_pair": [66666, 66667]
        },
        {
            "prompt": "Flat alpaca wool texture, silky fine fibers, uniform camel brown color, luxurious pile surface, material texture only, flat view, seamless texture",
            "filename_prefix": "alpaca_texture",
            "seed_pair": [77777, 77778]
        },
        {
            "prompt": "Flat cotton pile texture, dense cotton fibers, uniform white color, soft material surface, flat pile texture, top-down view, seamless pattern",
            "filename_prefix": "cotton_pile_texture",
            "seed_pair": [34567, 76543]
        },
        {
            "prompt": "Flat cotton dhurrie texture, tight cotton weave, uniform navy blue color, flat woven surface, material texture only, repeating pattern",
            "filename_prefix": "cotton_dhurrie_texture",
            "seed_pair": [45678, 87654]
        },
        {
            "prompt": "Flat cotton weave texture, tight plain weave, uniform white color, material surface only, clean textile texture, flat view, seamless tileable",
            "filename_prefix": "cotton_weave_texture",
            "seed_pair": [12300, 12301]
        },
        {
            "prompt": "Flat jute fiber texture, natural woven appearance, uniform beige color, coarse weave, material surface only, flat view, seamless tileable texture",
            "filename_prefix": "jute_texture",
        },
        {
            "prompt": "Flat jute boucle texture, large looped jute fibers, uniform tan color, nubby texture, natural material surface, flat appearance, repeating texture", # Added "large"
            "filename_prefix": "jute_boucle_texture",
            "seed_pair": [67890, 19876]
        },
        {
            "prompt": "Flat sisal fiber texture, tight natural weave, uniform light brown color, coarse texture, material surface only, flat view, seamless pattern",
            "filename_prefix": "sisal_texture",
            "seed_pair": [78901, 10987]
        },
        {
            "prompt": "Flat seagrass weave texture, natural fiber mat, uniform golden brown color, tight basketweave pattern, material surface only, flat view, seamless tileable",
            "filename_prefix": "seagrass_texture",
            "seed_pair": [11111, 11112]
        },
        {
            "prompt": "Flat coir fiber texture, coconut husk fibers, uniform dark brown color, coarse natural weave, material texture only, flat appearance, repeating pattern",
            "filename_prefix": "coir_texture",
            "seed_pair": [22222, 22223]
        },
        {
            "prompt": "Flat abaca fiber texture, banana plant fibers, uniform cream beige color, fine natural weave, material surface only, flat view, seamless texture",
            "filename_prefix": "abaca_texture",
            "seed_pair": [33333, 33334]
        },
        {
            "prompt": "Flat hemp fiber texture, natural plant fibers, uniform sage green color, sturdy weave pattern, material texture only, flat appearance, tileable pattern",
            "filename_prefix": "hemp_texture",
            "seed_pair": [44444, 44445]
        },
        {
            "prompt": "Flat nylon pile texture, dense synthetic fibers, uniform burgundy color, cut pile surface, material texture only, flat view, repeating pattern",
            "filename_prefix": "nylon_pile_texture",
        },
        {
            "prompt": "Flat polypropylene texture, synthetic fiber weave, uniform charcoal color, durable surface, material texture only, flat appearance, seamless texture",
            "filename_prefix": "polypropylene_texture",
            "seed_pair": [90123, 32109]
        },
        {
            "prompt": "Flat polyester shag texture, synthetic long thick fibers, uniform sage green color, deep pile surface, material texture only, flat view, tileable pattern", # Added "thick"
            "filename_prefix": "polyester_shag_texture",
            "seed_pair": [10234, 43210]
        },
        {
            "prompt": "Flat solution-dyed acrylic texture, synthetic outdoor fibers, uniform terracotta color, weather-resistant surface, material texture only, flat view, seamless repeating pattern",
            "filename_prefix": "acrylic_outdoor_texture",
            "seed_pair": [88888, 88889]
        },
        {
            "prompt": "Flat olefin fiber texture, synthetic marine-grade fibers, uniform teal blue color, moisture-resistant surface, material texture only, flat appearance, seamless texture",
            "filename_prefix": "olefin_texture",
            "seed_pair": [99999, 99990]
        },
        {
            "prompt": "Flat polyethylene texture, recycled synthetic fibers, uniform slate gray color, eco-friendly surface, material texture only, flat view, tileable pattern",
            "filename_prefix": "polyethylene_texture",
            "seed_pair": [10101, 10102]
        },
        {
            "prompt": "Flat viscose pile texture, silky synthetic fibers, uniform silver color, lustrous surface, material texture only, flat appearance, seamless repeating pattern",
            "filename_prefix": "viscose_pile_texture",
        },
        {
            "prompt": "Flat saxony pile texture, twisted cut fibers, uniform burgundy wine color, formal pile surface, material texture only, flat appearance, seamless texture",
            "filename_prefix": "saxony_pile_texture",
            "seed_pair": [20202, 20203]
        },
        {
            "prompt": "Flat plush pile texture, dense straight fibers, uniform midnight blue color, velvet-like surface, material texture only, flat view, tileable pattern",
            "filename_prefix": "plush_pile_texture",
            "seed_pair": [30303, 30304]
        },
        {
            "prompt": "Flat textured loop texture, multi-level large loops, uniform mushroom beige color, sculptured surface, material texture only, flat appearance, seamless repeating pattern", # Added "large"
            "filename_prefix": "textured_loop_texture",
            "seed_pair": [40404, 40405]
        },
        {
            "prompt": "Flat cut-and-loop texture, mixed pile heights with distinct patterns, uniform sage green color, dimensional surface, material texture only, flat view, seamless texture", # Added "distinct patterns"
            "filename_prefix": "cut_loop_texture",
            "seed_pair": [50505, 50506]
        },
        {
            "prompt": "Flat frieze texture, heat-set highly twisted fibers, uniform forest green color, durable knobby pile surface, material texture only, flat appearance, tileable pattern", # Added "highly", "knobby"
            "filename_prefix": "frieze_texture",
            "seed_pair": [65789, 98765]
        },
        {
            "prompt": "Flat flatweave texture, tight interlaced threads forming visible pattern, uniform rust orange color, smooth woven surface, material texture only, flat appearance, seamless repeating pattern", # Added "forming visible pattern"
            "filename_prefix": "flatweave_texture",
            "seed_pair": [60606, 60607]
        },
        {
            "prompt": "Flat twill weave texture, clear diagonal rib pattern, uniform charcoal black color, structured surface, material texture only, flat view, seamless texture", # Added "clear"
            "filename_prefix": "twill_weave_texture",
            "seed_pair": [70707, 70708]
        },
        {
            "prompt": "Flat herringbone weave texture, bold zigzag fiber pattern, uniform taupe color, classic woven surface, material texture only, flat appearance, tileable pattern", # Added "bold"
            "filename_prefix": "herringbone_texture",
            "seed_pair": [80808, 80809]
        },
        {
            "prompt": "Flat wool-nylon blend texture, mixed fiber pile, uniform taupe color, durable surface, material texture only, flat view, seamless repeating pattern",
            "filename_prefix": "wool_nylon_blend_texture",
            "seed_pair": [32456, 65432]
        },
        {
            "prompt": "Flat cotton-jute blend texture, mixed natural fibers with visible jute coarseness, uniform oatmeal color, textured weave, material surface only, flat appearance, seamless texture", # Added detail
            "filename_prefix": "cotton_jute_blend_texture",
            "seed_pair": [43567, 76542] 
        },
        {
            "prompt": "Flat wool-silk blend texture, mixed luxury fibers with silk sheen, uniform champagne color, lustrous pile surface, material texture only, flat view, tileable pattern", # Added detail
            "filename_prefix": "wool_silk_texture",
            "seed_pair": [90909, 90910]
        },
        {
            "prompt": "Flat linen-cotton blend texture, mixed natural fibers with characteristic linen slubs, uniform stone gray color, casual woven surface, material texture only, flat appearance, seamless repeating pattern", # Added detail
            "filename_prefix": "linen_cotton_texture",
            "seed_pair": [12121, 12122]
        },
        {
            "prompt": "Flat jute-wool blend texture, mixed textured fibers with coarse jute and soft wool, uniform sand beige color, rustic surface, material texture only, flat view, seamless texture", # Added detail
            "filename_prefix": "jute_wool_texture",
            "seed_pair": [23232, 23233]
        },
        {
            "prompt": "Flat chenille texture, chunky twisted yarn surface, uniform dusty blue color, plush material texture, soft pile, flat view, tileable pattern", # Added "chunky"
            "filename_prefix": "chenille_texture",
        }
    ]
    
    default_negative_prompt = inspect.signature(sd_generator.generate_two_variations).parameters['negative_prompt'].default
    print(f"Default negative prompt for SD generation: \"{default_negative_prompt}\"")

    for config in texture_prompts_configs:
        original_prompt = config.pop("prompt")
        enhanced_prompt = enhance_prompt_for_details(original_prompt)
        seeds_to_use = config.pop("seed_pair", None) 
        filename_prefix = config.get("filename_prefix", "texture")

        print(f"\nSD Gen: Processing prompt for '{filename_prefix}'...")
        generated_paths_for_prompt = sd_generator.generate_two_variations(
            prompt=enhanced_prompt, output_dir=output_directory,
            seeds=seeds_to_use, filename_prefix=filename_prefix
        )
        all_sd_image_paths_to_super_resolve.extend(generated_paths_for_prompt)

    print("\n--- PHASE 1 COMPLETE: All Stable Diffusion textures generated. ---")

    # --- Offload SD Model ---
    sd_generator.offload_sd_model()
    print("Waiting a few seconds for VRAM to clear...")
    time.sleep(5) 


    print("\n--- All Processing Complete ---")
