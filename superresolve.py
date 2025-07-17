import os
import sys
import subprocess
import shutil
import glob
import uuid
from pathlib import Path
import argparse
import time
import gc # Garbage Collector
if sys.platform == "win32":
    import ctypes # For calling Windows API functions

# --- Path Configurations ---
try:
    APP_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
except NameError:
    APP_ROOT = Path(os.getcwd()).resolve()

REPOS_DIR = APP_ROOT / "repositories"
MODELS_STORAGE_DIR = APP_ROOT / "pretrained_models_storage"

SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_DIR = REPOS_DIR / "SwinIR"

MODEL_URLS = {
    "SwinIR_L_x4_GAN.pth": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "path": MODELS_STORAGE_DIR / "swinir" / "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    }
}

TEMP_INPUT_DIR_BASE_FOR_BATCHES = APP_ROOT / "temp_swinir_batch_inputs" # Renamed for clarity
# TEMP_OUTPUT_DIR_BASE is not strictly needed as outputs go to user-specified folder

SR_MODEL_TYPE = "SwinIR-L"

# --- Helper Functions ---
def run_command(command_list, cwd=None, desc=None):
    if desc:
        print(desc)
    env = os.environ.copy()
    if cwd:
        python_path_addition = str(Path(cwd).resolve())
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{python_path_addition}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = python_path_addition
            
    process = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd), text=True, env=env)
    stdout, stderr = process.communicate() # Waits for process to complete

    if process.returncode != 0:
        print(f"Error executing command: {' '.join(command_list)}")
        print(f"Return code: {process.returncode}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
        raise RuntimeError(f"Command failed. Check console for details. Stderr: {stderr or stdout}")
    return stdout, stderr

# --- Setup Function ---
def setup_environment():
    print("Performing environment setup for SwinIR...")
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_INPUT_DIR_BASE_FOR_BATCHES.mkdir(parents=True, exist_ok=True)

    if not SWINIR_DIR.exists():
        print(f"Cloning SwinIR into {SWINIR_DIR}...")
        run_command(["git", "clone", SWINIR_REPO_URL, str(SWINIR_DIR)])
        print("SwinIR cloned. Ensure its dependencies are installed (e.g., `pip install basicsr timm opencv-python`).")
    else:
        print(f"SwinIR repository found at {SWINIR_DIR}.")

    for model_name, model_info in MODEL_URLS.items():
        model_path = Path(model_info["path"])
        if not model_path.exists():
            print(f"Downloading {model_name} to {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                run_command(["wget", model_info["url"], "-O", str(model_path)])
            except (FileNotFoundError, RuntimeError):
                print("wget failed or not found, trying curl...")
                try:
                    run_command(["curl", "-L", model_info["url"], "-o", str(model_path), "--create-dirs"])
                except Exception as e:
                    print(f"CRITICAL: Failed to download {model_name} using both wget and curl. Error: {e}")
                    raise
        else:
            print(f"SwinIR model {model_name} found at {model_path}.")
    print("SwinIR setup check complete.")

# --- SwinIR Batch Inference Function ---
def process_batch_with_swinir(batch_image_paths: list, output_folder: Path, model_type: str, test_patch_wise: bool, tile_size: str):
    """
    Processes a batch of images with SwinIR.
    SwinIR's script reads from a temporary input folder and saves to its own results folder.
    This function then moves the results to the specified final output_folder.
    """
    if not batch_image_paths:
        return

    batch_id = str(uuid.uuid4())
    # Create a unique temporary input directory for this specific batch
    # This directory will contain copies of the images for SwinIR to process
    current_batch_temp_input_dir = TEMP_INPUT_DIR_BASE_FOR_BATCHES / batch_id
    current_batch_temp_input_dir.mkdir(parents=True, exist_ok=True)

    print(f"SwinIR Batch {batch_id}: Preparing {len(batch_image_paths)} images...")
    for img_path in batch_image_paths:
        shutil.copy(str(img_path), str(current_batch_temp_input_dir / img_path.name))

    script_path = SWINIR_DIR / "main_test_swinir.py"
    if model_type == "SwinIR-L":
        model_file = Path(MODEL_URLS["SwinIR_L_x4_GAN.pth"]["path"])
        # This is the subfolder SwinIR-L (large model) saves into within its 'results' dir
        swinir_output_subfolder_in_repo = "swinir_real_sr_x4_large" 
    else:
        raise ValueError(f"Unsupported SwinIR model type: {model_type}. Expected SwinIR-L.")

    if not model_file.exists():
        raise FileNotFoundError(f"SwinIR model file not found: {model_file}. Run setup or check paths.")

    # SwinIR will save its results into SWINIR_DIR / "results" / swinir_output_subfolder_in_repo
    # We clean this before each batch run to ensure it's fresh for this batch's outputs.
    swinir_repo_results_path = SWINIR_DIR / "results"
    if swinir_repo_results_path.exists():
        shutil.rmtree(swinir_repo_results_path)
    # SwinIR's script will recreate SWINIR_DIR / "results" and the subfolder

    cmd = [
        sys.executable, str(script_path),
        "--task", "real_sr", "--model_path", str(model_file),
        "--folder_lq", str(current_batch_temp_input_dir), # SwinIR processes all images in this folder
        "--scale", "4"
    ]
    if model_type == "SwinIR-L": cmd.append("--large_model")
    if test_patch_wise: cmd.extend(["--tile", tile_size])
    
    run_command(cmd, cwd=SWINIR_DIR, desc=f"SwinIR Batch {batch_id}: Running {model_type} on {len(batch_image_paths)} images (patch-wise: {test_patch_wise})...")

    # Move processed images from SwinIR's results folder to the final output_folder
    # The expected location of outputs from SwinIR's script:
    swinir_actual_output_dir = swinir_repo_results_path / swinir_output_subfolder_in_repo
    
    moved_count = 0
    if swinir_actual_output_dir.exists():
        for original_img_path in batch_image_paths:
            # SwinIR usually appends "_SwinIR" to the filename before the extension
            # e.g., input "image.png" becomes "image_SwinIR.png"
            sr_filename_stem = f"{original_img_path.stem}_SwinIR"
            
            # Try to find the output file, could be .png or other common extensions if SwinIR changes behavior
            found_sr_file = None
            for ext_glob in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]: # Common image extensions
                possible_files = list(swinir_actual_output_dir.glob(f"{sr_filename_stem}{ext_glob[1:]}")) # ext_glob[1:] to remove '*'
                if possible_files:
                    found_sr_file = possible_files[0]
                    break
            
            if found_sr_file and found_sr_file.exists():
                # Final output name in the user-specified folder
                final_output_filename = f"{original_img_path.stem}_swinir_x4{original_img_path.suffix}"
                final_output_path = output_folder / final_output_filename
                try:
                    shutil.move(str(found_sr_file), str(final_output_path))
                    # print(f"Moved {found_sr_file.name} to {final_output_path}")
                    moved_count +=1
                except Exception as e_move:
                    print(f"Error moving {found_sr_file.name} to {final_output_path}: {e_move}")
            else:
                print(f"Warning: Output for {original_img_path.name} (expected like {sr_filename_stem}.png) not found in {swinir_actual_output_dir}")
        print(f"SwinIR Batch {batch_id}: Moved {moved_count} processed images to {output_folder}.")
    else:
        print(f"Warning: SwinIR output directory {swinir_actual_output_dir} not found after processing batch {batch_id}.")


    # Clean up the temporary input directory for this batch
    if current_batch_temp_input_dir.exists():
        shutil.rmtree(current_batch_temp_input_dir)
        # print(f"Cleaned up temporary batch input directory: {current_batch_temp_input_dir}")
    
    # Clean up SwinIR's own results folder again to be safe, if it still exists
    if swinir_repo_results_path.exists():
        shutil.rmtree(swinir_repo_results_path, ignore_errors=True)
        # Give Windows a moment if many files were involved
        if sys.platform == "win32": time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Batch super-resolve images in a folder using SwinIR-L x4.")
    parser.add_argument("input_folder", type=Path, help="Path to the folder containing images to super-resolve.")
    parser.add_argument("output_folder", type=Path, help="Path to the folder where super-resolved images will be saved.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of images to process in a single SwinIR subprocess call. Reduce if crashing.")
    parser.add_argument("--patch_wise", action="store_true", help="Enable patch-wise processing (tile mode) for low VRAM within SwinIR.")
    parser.add_argument("--tile_size", type=str, default="640", help="Tile size for patch-wise processing (e.g., '640').")
    parser.add_argument("--delay_between_batches", type=float, default=5.0, help="Seconds to wait between processing batches to allow system to recover.")


    args = parser.parse_args()

    if not args.input_folder.is_dir():
        print(f"Error: Input folder not found: {args.input_folder}")
        sys.exit(1)

    args.output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Super-resolved images will be saved to: {args.output_folder}")
    print(f"Processing with Batch Size: {args.batch_size}, Patch-wise: {args.patch_wise}, Tile Size: {args.tile_size}, Delay: {args.delay_between_batches}s")


    try:
        setup_environment()
    except Exception as e:
        print(f"CRITICAL: SwinIR Environment setup failed: {e}. Cannot proceed.")
        sys.exit(1)

    supported_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(args.input_folder.glob(f"*{ext}")))
        image_files.extend(list(args.input_folder.glob(f"*{ext.upper()}")))

    if not image_files:
        print(f"No supported image files found in {args.input_folder}.")
        sys.exit(0)
        
    image_files = sorted(list(set(image_files)))

    print(f"\nFound {len(image_files)} images in '{args.input_folder}'.")


    # Filter out images that already have super-resolved versions
    images_to_process = []
    skipped_count = 0
    
    for img_path in image_files:
        sr_filename = f"{img_path.stem}_swinir_x4{img_path.suffix}"
        sr_path = args.output_folder / sr_filename
        
        if sr_path.exists():
            skipped_count += 1
            print(f"SKIP: {sr_filename} already exists")
        else:
            images_to_process.append(img_path)
    
    print(f"\nFiltered results: {len(images_to_process)} images to process, {skipped_count} already processed")
    
    if not images_to_process:
        print("All images already have super-resolved versions. Nothing to do.")
        sys.exit(0)
    
    # Recalculate batch info based on filtered list
    total_batches = (len(images_to_process) + args.batch_size - 1) // args.batch_size
    print(f"Processing {len(images_to_process)} images in {total_batches} batches.")

    for i in range(0, len(images_to_process), args.batch_size):
        batch_image_files = images_to_process[i:i + args.batch_size]
        current_batch_num = (i // args.batch_size) + 1
        print(f"\n--- Processing Batch {current_batch_num} of {total_batches} ---")
        
        try:
            process_batch_with_swinir(
                batch_image_files,
                args.output_folder,
                model_type=SR_MODEL_TYPE,
                test_patch_wise=args.patch_wise,
                tile_size=args.tile_size
            )
        except Exception as e:
            print(f"Error processing batch starting with {batch_image_files[0].name if batch_image_files else 'N/A'}: {e}")
            import traceback
            traceback.print_exc()
            if "out of memory" in str(e).lower():
                 print("SwinIR: OOM detected. Try a smaller --batch_size and/or ensure --patch_wise is used.")
            print("Attempting to continue with the next batch...")
        
        # --- Memory Cleanup and Delay after each batch ---
        print(f"\n--- Batch {current_batch_num} processed. Attempting memory cleanup and delay ---")
        
        # 1. Python's Garbage Collection
        gc.collect()
        print("Python garbage collection called.")

        # 2. Windows-specific heap minimization (best effort)
        if sys.platform == "win32":
            try:
                # msvcrt.dll is the Microsoft Visual C++ Runtime Library
                # _heapmin() attempts to shrink the C runtime heap
                msvcrt = ctypes.CDLL("msvcrt.dll")
                msvcrt._heapmin()
                print("Windows: Called msvcrt._heapmin() to attempt C runtime heap minimization.")
            except Exception as e_heapmin:
                print(f"Windows: Could not call _heapmin(): {e_heapmin}")
        
        # 3. Delay
        if current_batch_num < total_batches: # No delay after the last batch
            print(f"Waiting for {args.delay_between_batches} seconds before next batch...")
            time.sleep(args.delay_between_batches)

    print("\nBatch super-resolution process complete.")
    
    # Final cleanup of the base temporary input directory if it's empty
    try:
        if TEMP_INPUT_DIR_BASE_FOR_BATCHES.exists() and not any(TEMP_INPUT_DIR_BASE_FOR_BATCHES.iterdir()):
            shutil.rmtree(TEMP_INPUT_DIR_BASE_FOR_BATCHES)
            print(f"Cleaned up empty base temporary input directory: {TEMP_INPUT_DIR_BASE_FOR_BATCHES}")
        elif TEMP_INPUT_DIR_BASE_FOR_BATCHES.exists():
            print(f"Note: Base temporary input directory {TEMP_INPUT_DIR_BASE_FOR_BATCHES} may still contain subfolders if errors occurred during processing.")
    except Exception as e_cleanup:
        print(f"Error during final cleanup of temp base directory: {e_cleanup}")


if __name__ == "__main__":
    main()