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
            "prompt": "Complete wool pile rug texture, dense natural wool fibers creating a minutely stippled surface, uniform cream color, soft full pile appearance, material texture only, flat top-down view, seamless repeating pattern, macro photography, individual fiber ends resolve on 4x magnification.",
            "filename_prefix": "rug_wool_pile_cream_micropile",
            "seed_pair": [12345, 54321]
        },
        {
            "prompt": "Full wool loop pile rug texture, uniform gray wool loops forming a dense berber-style surface with a subtle particulate quality, material texture only, flat appearance, seamless texture, macro detail, loop structure sharpens with 4x SR.",
            "filename_prefix": "rug_wool_loop_gray_berber",
            # Random seeds
        },
        {
            "prompt": "Area rug texture of fine merino wool, soft dense fibers creating an almost smooth but microscopically granular surface, uniform ivory color, material texture only, flat view, seamless tileable, fine fiber details emerge at 4x magnification.",
            "filename_prefix": "rug_merino_ivory_granular",
            "seed_pair": [55555, 55556]
        },
        {
            "prompt": "Complete lambswool rug texture, curly soft fibers forming a fluffy yet dense pile with a fine particulate surface, uniform pearl gray color, material texture only, flat appearance, repeating pattern, macro shot, curly fiber details resolve distinctly at 4x SR.",
            "filename_prefix": "rug_lambswool_pearlgray_particulate",
            "seed_pair": [66666, 66667]
        },
        {
            "prompt": "Full alpaca wool rug texture, silky fine fibers creating a luxurious, minutely textured pile, uniform camel brown color, material texture only, flat top-down view, seamless texture, high detail, silky fiber sheen and micro-texture enhance at 4x magnification.",
            "filename_prefix": "rug_alpaca_camel_microtexture",
            "seed_pair": [77777, 77778]
        },

        # --- Cotton Rugs ---
        {
            "prompt": "Complete cotton pile rug texture, dense cotton fibers forming a soft, minutely stippled surface, uniform white color, full pile appearance, material texture only, flat top-down view, seamless pattern, individual cotton fiber ends resolve on 4x magnification.",
            "filename_prefix": "rug_cotton_pile_white_stippled",
            "seed_pair": [34567, 76543]
        },
        {
            "prompt": "Full cotton dhurrie rug texture, tight cotton weave creating a flat woven surface with subtle particulate weave detail, uniform navy blue color, material texture only, repeating pattern, macro photography, fine weave structure clarifies at 4x SR.",
            "filename_prefix": "rug_cotton_dhurrie_navy_particulate",
            "seed_pair": [45678, 87654]
        },
        {
            "prompt": "Area rug texture of tight plain cotton weave, uniform white color, presenting a clean textile surface with a microscopic granular quality, material texture only, flat view, seamless tileable, fine thread details enhance with 4x magnification.",
            "filename_prefix": "rug_cotton_weave_white_granular",
            "seed_pair": [12300, 12301]
        },

        # --- Natural Fiber Rugs (Jute, Sisal, etc.) ---
        {
            "prompt": "Complete jute fiber rug texture, natural woven appearance with a coarse yet minutely detailed particulate surface, uniform beige color, material texture only, flat top-down view, seamless tileable texture, fiber irregularities resolve distinctly at 4x SR.",
            "filename_prefix": "rug_jute_beige_particulate",
            # Random seeds
        },
        {
            "prompt": "Full jute boucle rug texture, large looped jute fibers forming a nubby, textured surface with underlying fine particulate detail, uniform tan color, material texture only, flat appearance, repeating texture, macro detail, loop and fiber details sharpen at 4x magnification.",
            "filename_prefix": "rug_jute_boucle_tan_particulate",
            "seed_pair": [67890, 19876]
        },
        {
            "prompt": "Area rug texture of tight natural sisal weave, uniform light brown color, coarse yet minutely granular surface appearance, material texture only, flat view, seamless pattern, individual sisal fiber details emerge at 4x SR.",
            "filename_prefix": "rug_sisal_lightbrown_granular",
            "seed_pair": [78901, 10987]
        },
        {
            "prompt": "Complete seagrass rug texture, natural fiber mat woven into a tight basketweave with a subtle particulate surface, uniform golden brown color, material texture only, flat top-down view, seamless tileable, weave and fiber details resolve clearly at 4x magnification.",
            "filename_prefix": "rug_seagrass_goldenbrown_particulate",
            "seed_pair": [11111, 11112]
        },
        {
            "prompt": "Full coir rug texture, coarse coconut husk fibers forming a natural weave with a dense particulate quality, uniform dark brown color, material texture only, flat appearance, repeating pattern, macro shot, individual coir fiber details for 4x SR enhancement.",
            "filename_prefix": "rug_coir_darkbrown_particulate",
            "seed_pair": [22222, 22223]
        },
        {
            "prompt": "Area rug texture of fine natural abaca weave, banana plant fibers, uniform cream beige color, presenting a smooth yet microscopically fibrous surface, material texture only, flat view, seamless texture, delicate fiber network resolves on 4x magnification.",
            "filename_prefix": "rug_abaca_creambeige_microfibrous",
            "seed_pair": [33333, 33334]
        },
        {
            "prompt": "Complete hemp fiber rug texture, sturdy natural plant fibers woven into a surface with a fine particulate texture, uniform sage green color, material texture only, flat appearance, tileable pattern, macro focus, woven fiber details sharpen at 4x SR.",
            "filename_prefix": "rug_hemp_sagegreen_particulate",
            "seed_pair": [44444, 44445]
        },

        # --- Synthetic Fiber Rugs ---
        {
            "prompt": "Full nylon pile rug texture, dense synthetic fibers creating a minutely stippled cut pile surface, uniform burgundy color, material texture only, flat top-down view, repeating pattern, macro photography, individual fiber ends resolve clearly at 4x magnification.",
            "filename_prefix": "rug_nylon_pile_burgundy_stippled",
            # Random seeds
        },
        {
            "prompt": "Area rug texture of polypropylene synthetic fiber weave, uniform charcoal color, forming a durable surface with a subtle particulate quality, material texture only, flat appearance, seamless texture, fine weave detail emerges with 4x SR.",
            "filename_prefix": "rug_polypropylene_charcoal_particulate",
            "seed_pair": [90123, 32109]
        },
        {
            "prompt": "Complete polyester shag rug texture, synthetic long thick fibers creating a deep pile with an underlying dense particulate base, uniform sage green color, material texture only, flat view, tileable pattern, macro shot, individual shag fibers and base details for 4x SR.",
            "filename_prefix": "rug_polyester_shag_sagegreen_particulate",
            "seed_pair": [10234, 43210]
        },
        {
            "prompt": "Full solution-dyed acrylic outdoor rug texture, synthetic fibers creating a weather-resistant surface with a fine granular texture, uniform terracotta color, material texture only, flat top-down view, seamless repeating pattern, micro-texture details resolve on 4x magnification.",
            "filename_prefix": "rug_acrylic_outdoor_terracotta_granular",
            "seed_pair": [88888, 88889]
        },
        {
            "prompt": "Area rug texture of olefin marine-grade fibers, uniform teal blue color, creating a moisture-resistant surface with a minutely stippled appearance, material texture only, flat appearance, seamless texture, fine fiber points sharpen at 4x SR.",
            "filename_prefix": "rug_olefin_teal_stippled",
            "seed_pair": [99999, 99990]
        },
        {
            "prompt": "Complete polyethylene recycled fiber rug texture, uniform slate gray color, forming an eco-friendly surface with a dense particulate micro-texture, material texture only, flat view, tileable pattern, macro photography, details emerge clearly with 4x magnification.",
            "filename_prefix": "rug_polyethylene_slategray_particulate",
            "seed_pair": [10101, 10102]
        },
        {
            "prompt": "Full viscose pile rug texture, silky synthetic fibers creating a lustrous surface with a very fine, almost imperceptible granular quality, uniform silver color, material texture only, flat appearance, seamless repeating pattern, silky sheen and micro-granules enhance at 4x SR.",
            "filename_prefix": "rug_viscose_silver_microgranular",
            # Random seeds
        },

        # --- Specialty Pile Rugs ---
        {
            "prompt": "Area rug texture of saxony pile, twisted cut fibers forming a formal pile surface with a minutely textured appearance, uniform burgundy wine color, material texture only, flat appearance, seamless texture, details of twisted fibers resolve on 4x magnification.",
            "filename_prefix": "rug_saxony_burgundywine_microtextured",
            "seed_pair": [20202, 20203]
        },
        {
            "prompt": "Complete plush pile rug texture, dense straight fibers creating a velvet-like surface with a subtle particulate quality, uniform midnight blue color, material texture only, flat top-down view, tileable pattern, macro shot, individual plush fiber ends sharpen at 4x SR.",
            "filename_prefix": "rug_plush_midnightblue_particulate",
            "seed_pair": [30303, 30304]
        },
        {
            "prompt": "Full textured loop rug texture, multi-level large loops forming a sculptured surface with an underlying fine particulate base, uniform mushroom beige color, material texture only, flat appearance, seamless repeating pattern, loop and base details enhance with 4x magnification.",
            "filename_prefix": "rug_texturedloop_mushroom_particulate",
            "seed_pair": [40404, 40405]
        },
        {
            "prompt": "Area rug texture of cut-and-loop pile, mixed pile heights with distinct patterns formed by minutely textured loops and cut pile, uniform sage green color, dimensional surface, material texture only, flat view, seamless texture, pattern details resolve clearly at 4x SR.",
            "filename_prefix": "rug_cutloop_sagegreen_microtextured",
            "seed_pair": [50505, 50506]
        },
        {
            "prompt": "Complete frieze rug texture, heat-set highly twisted fibers creating a durable knobby pile surface with a fine particulate underlying texture, uniform forest green color, material texture only, flat appearance, tileable pattern, macro focus, twisted fiber details for 4x SR enhancement.",
            "filename_prefix": "rug_frieze_forestgreen_particulate",
            "seed_pair": [65789, 98765]
        },

        # --- Specialty Weave Rugs ---
        {
            "prompt": "Full flatweave rug texture, tight interlaced threads forming a visible pattern with a smooth yet microscopically granular woven surface, uniform rust orange color, material texture only, flat appearance, seamless repeating pattern, fine thread details resolve distinctly on 4x magnification.",
            "filename_prefix": "rug_flatweave_rustorange_granular",
            "seed_pair": [60606, 60607]
        },
        {
            "prompt": "Area rug texture of twill weave, clear diagonal rib pattern creating a structured surface with a subtle particulate quality between ribs, uniform charcoal black color, material texture only, flat view, seamless texture, diagonal weave details sharpen at 4x SR.",
            "filename_prefix": "rug_twill_charcoal_particulate",
            "seed_pair": [70707, 70708]
        },
        {
            "prompt": "Complete herringbone weave rug texture, bold zigzag fiber pattern forming a classic woven surface with a minutely textured feel, uniform taupe color, material texture only, flat appearance, tileable pattern, macro shot, zigzag weave and fiber details for 4x magnification.",
            "filename_prefix": "rug_herringbone_taupe_microtextured",
            "seed_pair": [80808, 80809]
        },

        # --- Blended Material Rugs ---
        {
            "prompt": "Full wool-nylon blend rug texture, mixed fiber pile creating a durable surface with a dense, minutely stippled appearance, uniform taupe color, material texture only, flat top-down view, seamless repeating pattern, individual mixed fiber ends resolve on 4x magnification.",
            "filename_prefix": "rug_woolnylon_taupe_stippled",
            "seed_pair": [32456, 65432]
        },
        {
            "prompt": "Area rug texture of cotton-jute blend, mixed natural fibers with visible jute coarseness within a textured weave that has an overall fine particulate quality, uniform oatmeal color, material texture only, flat appearance, seamless texture, mixed fiber details enhance at 4x SR.",
            "filename_prefix": "rug_cottonjute_oatmeal_particulate",
            "seed_pair": [43567, 76542]
        },
        {
            "prompt": "Complete wool-silk blend rug texture, mixed luxury fibers creating a lustrous pile surface with silk sheen and a microscopic granular texture, uniform champagne color, material texture only, flat view, tileable pattern, sheen and micro-granules for 4x magnification.",
            "filename_prefix": "rug_woolsilk_champagne_granular",
            "seed_pair": [90909, 90910]
        },
        {
            "prompt": "Full linen-cotton blend rug texture, mixed natural fibers with characteristic linen slubs within a casual woven surface that presents a subtle particulate detail, uniform stone gray color, material texture only, flat appearance, seamless repeating pattern, slub and weave details resolve on 4x magnification.",
            "filename_prefix": "rug_linencotton_stonegray_particulate",
            "seed_pair": [12121, 12122]
        },
        {
            "prompt": "Area rug texture of jute-wool blend, mixed textured fibers with coarse jute and soft wool forming a rustic surface with an underlying fine particulate quality, uniform sand beige color, material texture only, flat view, seamless texture, details of mixed fibers enhance at 4x SR.",
            "filename_prefix": "rug_jutewool_sandbeige_particulate",
            "seed_pair": [23232, 23233]
        },

        # --- Specialty Material Rugs (e.g., Chenille) ---
        {
            "prompt": "Complete chenille rug texture, chunky twisted yarn surface creating a plush material texture with a soft pile that has a minutely detailed particulate base between yarns, uniform dusty blue color, material texture only, flat top-down view, tileable pattern, macro focus, yarn and base details for 4x magnification.",
            "filename_prefix": "rug_chenille_dustyblue_particulate",
            # Random seeds
        },

        # --- Prompts based on the "gray example" but with varied colors ---
        {
            "prompt": "Complete micro-fiber rug texture, ultra-dense synthetic fibers creating a particulate surface appearance, uniform deep forest green color, material texture only, flat top-down view, seamless repeating pattern, macro photography, details resolve to distinct micro-points on 4x magnification.",
            "filename_prefix": "rug_microfiber_dense_forestgreen",
            "seed_pair": [62001, 62002]
        },
        {
            "prompt": "Full short pile area rug texture, tightly compacted mono-filament fibers forming a minutely stippled surface, uniform rich burgundy color, material texture only, flat appearance, seamless texture, high detail, becomes distinct micro-points when super-resolved.",
            "filename_prefix": "rug_shortpile_stippled_burgundy",
            # Random seeds
        },
        {
            "prompt": "Area rug texture of felted industrial textile, extremely dense non-woven fibers creating a fine granular matte surface, uniform warm terracotta color, material texture only, top-down flat lay, seamless tileable, photorealistic, intended for 4x super-resolution revealing micro-fibers.",
            "filename_prefix": "rug_felted_granular_terracotta",
            "seed_pair": [62005, 62006]
        },
        {
            "prompt": "Complete dense flocking material rug texture, countless ultra-short vertical fibers creating a velvety yet particulate surface, uniform muted teal color, material texture only, flat top-down view, seamless texture, pre-super-resolution detail, sharpens to micro-dots at 4x SR.",
            "filename_prefix": "rug_flocking_particulate_mutedteal",
            # Random seeds
        },
        {
            "prompt": "Full synthetic suede alternative rug texture, micro-denier fibers forming a surface that appears almost smooth but with a microscopic granular quality, uniform ochre yellow color, material texture only, flat appearance, seamless repeating pattern, subtle nap details enhance at 4x magnification.",
            "filename_prefix": "rug_microsuede_granular_ochre",
            "seed_pair": [62011, 62012]
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
