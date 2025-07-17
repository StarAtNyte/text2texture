#!/usr/bin/env python3

import os
import sys
import subprocess
import glob
import time
import shutil
from pathlib import Path
import argparse
from PIL import Image
import torch

# --- Configuration ---
try:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_ROOT = os.getcwd()

REPOS_DIR = os.path.join(APP_ROOT, "repositories")
MODELS_STORAGE_DIR = os.path.join(APP_ROOT, "pretrained_models_storage")
SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
SWINIR_DIR = os.path.join(REPOS_DIR, "SwinIR")

# Updated model configuration to match the checkpoint
MODEL_URLS = {
    "SwinIR_L_x4_GAN.pth": {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "path": os.path.join(MODELS_STORAGE_DIR, "swinir", "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
    }
}

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
        return False, stdout, stderr
    return True, stdout, stderr

def setup_swinir_environment():
    """Setup SwinIR environment and download models if needed."""
    print("Setting up SwinIR environment...")
    os.makedirs(REPOS_DIR, exist_ok=True)
    os.makedirs(MODELS_STORAGE_DIR, exist_ok=True)
    
    # Clone SwinIR repository if not exists
    if not os.path.exists(SWINIR_DIR):
        print(f"Cloning SwinIR into {SWINIR_DIR}...")
        success, stdout, stderr = run_command(["git", "clone", SWINIR_REPO_URL, SWINIR_DIR])
        if not success:
            raise RuntimeError(f"Failed to clone SwinIR: {stderr}")
        print("SwinIR cloned successfully.")
    else:
        print(f"SwinIR repository found at {SWINIR_DIR}.")
    
    # Download models if not exists
    for model_name, model_info in MODEL_URLS.items():
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} to {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            success = False
            try:
                success, stdout, stderr = run_command(["wget", model_info["url"], "-O", model_path])
                if not success:
                    raise RuntimeError("wget failed")
            except (FileNotFoundError, RuntimeError):
                print("wget failed, trying curl...")
                success, stdout, stderr = run_command(["curl", "-L", model_info["url"], "-o", model_path, "--create-dirs"])
                if not success:
                    raise RuntimeError(f"Failed to download model: {stderr}")
        else:
            print(f"SwinIR model {model_name} found at {model_path}.")
    
    print("SwinIR setup complete.")

def get_image_files(input_dir, extensions=None):
    """Get all image files from input directory."""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    
    image_files = []
    input_path = Path(input_dir)
    
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)

def simple_upscale_fallback(image_path, output_path, scale=4):
    """Simple fallback upscaling using PIL."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            new_size = (width * scale, height * scale)
            upscaled = img.resize(new_size, Image.LANCZOS)
            upscaled.save(output_path)
        return True
    except Exception as e:
        print(f"  Fallback upscaling failed for {image_path}: {e}")
        return False

def process_batch_swinir(image_batch, output_dir, temp_input_dir, temp_output_dir, model_path, batch_num, total_batches):
    """Process a batch of images through SwinIR with corrected parameters."""
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(image_batch)} images) with SwinIR...")
    
    # Clear temp directories
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Copy images to temp input directory
    temp_image_paths = []
    for img_path in image_batch:
        temp_img_path = os.path.join(temp_input_dir, img_path.name)
        shutil.copy2(img_path, temp_img_path)
        temp_image_paths.append(temp_img_path)
        print(f"  Copied: {img_path.name}")
    
    # Try SwinIR with corrected parameters for the specific model
    try:
        swinir_main_script = os.path.join(SWINIR_DIR, "main_test_swinir.py")
        
        # Use parameters that match the downloaded model
        cmd = [
            sys.executable, swinir_main_script,
            "--task", "real_sr",
            "--scale", "4",
            "--large_model",  # This is important for the L model
            "--model_path", model_path,
            "--folder_lq", temp_input_dir,
            "--tile", "640",
            "--tile_overlap", "32"
        ]
        
        print(f"  Running SwinIR command: {' '.join(cmd)}")
        success, stdout, stderr = run_command(cmd, cwd=SWINIR_DIR, desc=f"  Processing batch {batch_num} with SwinIR...")
        
        if not success:
            print(f"  SwinIR failed: {stderr}")
            return False
        
        # Find and move output images
        results_dir = os.path.join(SWINIR_DIR, "results")
        processed_count = 0
        
        if os.path.exists(results_dir):
            # Look for output images in results directory and subdirectories
            for root, dirs, files in os.walk(results_dir):
                for output_file in files:
                    if output_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_path = os.path.join(root, output_file)
                        
                        # Create output filename
                        base_name, ext = os.path.splitext(output_file)
                        if not base_name.endswith('_SwinIR'):
                            base_name = base_name + '_SwinIR'
                        dst_path = os.path.join(output_dir, f"{base_name}{ext}")
                        
                        # Handle filename conflicts
                        counter = 1
                        while os.path.exists(dst_path):
                            dst_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                            counter += 1
                        
                        shutil.move(src_path, dst_path)
                        print(f"  Output saved: {os.path.basename(dst_path)}")
                        processed_count += 1
        
        # Clean up SwinIR results directory
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        
        return processed_count > 0
            
    except Exception as e:
        print(f"  ERROR processing batch {batch_num} with SwinIR: {e}")
        return False

def process_batch_fallback(image_batch, output_dir, batch_num, total_batches):
    """Process a batch using simple upscaling fallback."""
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(image_batch)} images) with fallback method...")
    
    success_count = 0
    for img_path in image_batch:
        base_name, ext = os.path.splitext(img_path.name)
        output_name = f"{base_name}_upscaled{ext}"
        output_path = os.path.join(output_dir, output_name)
        
        # Handle filename conflicts
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base_name}_upscaled_{counter}{ext}")
            counter += 1
        
        if simple_upscale_fallback(img_path, output_path):
            print(f"  Fallback success: {output_name}")
            success_count += 1
        else:
            print(f"  Fallback failed: {img_path.name}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Super-resolve existing images using SwinIR with fallback')
    parser.add_argument('input_dir', help='Directory containing images to super-resolve')
    parser.add_argument('output_dir', help='Directory to save super-resolved images')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of images to process per batch (default: 5)')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between batches in seconds (default: 2.0)')
    parser.add_argument('--extensions', nargs='+', default=['.png', '.jpg', '.jpeg'], help='Image extensions to process')
    parser.add_argument('--fallback_only', action='store_true', help='Skip SwinIR and use only simple upscaling')
    parser.add_argument('--no_fallback', action='store_true', help='Don\'t use fallback method if SwinIR fails')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup SwinIR environment (unless fallback only)
    model_path = None
    if not args.fallback_only:
        try:
            setup_swinir_environment()
            model_path = MODEL_URLS["SwinIR_L_x4_GAN.pth"]["path"]
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                if args.no_fallback:
                    sys.exit(1)
                else:
                    print("Will use fallback method instead.")
                    args.fallback_only = True
        except Exception as e:
            print(f"Error setting up SwinIR environment: {e}")
            if args.no_fallback:
                sys.exit(1)
            else:
                print("Will use fallback method instead.")
                args.fallback_only = True
    
    # Get all image files
    image_files = get_image_files(args.input_dir, args.extensions)
    if not image_files:
        print(f"No image files found in '{args.input_dir}' with extensions {args.extensions}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process.")
    
    if args.fallback_only:
        print("Using fallback method (simple upscaling) for all images.")
    else:
        print("Will attempt SwinIR first, with fallback if needed.")
    
    # Create temp directories
    temp_input_dir = os.path.join(APP_ROOT, "temp_sr_input")
    temp_output_dir = os.path.join(APP_ROOT, "temp_sr_output")
    
    # Process images in batches
    total_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
    successful_batches = 0
    swinir_successes = 0
    fallback_successes = 0
    
    for i in range(0, len(image_files), args.batch_size):
        batch_num = (i // args.batch_size) + 1
        batch = image_files[i:i + args.batch_size]
        
        batch_success = False
        
        if not args.fallback_only:
            # Try SwinIR first
            try:
                batch_success = process_batch_swinir(batch, args.output_dir, temp_input_dir, temp_output_dir, model_path, batch_num, total_batches)
                if batch_success:
                    swinir_successes += 1
            except Exception as e:
                print(f"Error processing batch {batch_num} with SwinIR: {e}")
                batch_success = False
        
        # Use fallback if SwinIR failed or fallback_only is set
        if not batch_success and not args.no_fallback:
            try:
                batch_success = process_batch_fallback(batch, args.output_dir, batch_num, total_batches)
                if batch_success:
                    fallback_successes += 1
            except Exception as e:
                print(f"Error processing batch {batch_num} with fallback: {e}")
                batch_success = False
        
        if batch_success:
            successful_batches += 1
        else:
            print(f"Batch {batch_num} completely failed.")
        
        # Delay between batches
        if batch_num < total_batches and args.delay > 0:
            print(f"Waiting {args.delay} seconds before next batch...")
            time.sleep(args.delay)
    
    # Clean up temp directories
    for temp_dir in [temp_input_dir, temp_output_dir]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print(f"\n--- Super-resolution Complete ---")
    print(f"Total batches: {total_batches}")
    print(f"Successful batches: {successful_batches}")
    print(f"SwinIR successes: {swinir_successes}")
    print(f"Fallback successes: {fallback_successes}")
    print(f"Output images saved to: {args.output_dir}")
    
    # Count output files
    output_files = get_image_files(args.output_dir)
    print(f"Total output images: {len(output_files)}")

if __name__ == "__main__":
    main()