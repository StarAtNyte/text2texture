#!/usr/bin/env python3
"""
PNG to WebP Converter Script
Converts all PNG images in a specified folder to compressed WebP format.
"""

import os
import sys
from PIL import Image
import argparse
from pathlib import Path

def convert_png_to_webp(input_folder, output_folder=None, quality=80, lossless=False, recursive=False):
    """
    Convert all PNG images in a folder to WebP format.
    
    Args:
        input_folder (str): Path to the folder containing PNG images
        output_folder (str): Path to the folder where WebP files will be saved
        quality (int): Compression quality (0-100, only for lossy compression)
        lossless (bool): Use lossless compression
        recursive (bool): Process subfolders recursively
    """
    
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Folder '{input_folder}' does not exist.")
        return
    
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a directory.")
        return
    
    # Set up output folder
    if output_folder is None:
        output_folder = input_folder + "_webp"
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    # Find all PNG files
    pattern = "**/*.png" if recursive else "*.png"
    png_files = list(input_path.glob(pattern))
    
    if not png_files:
        print(f"No PNG files found in '{input_folder}'")
        return
    
    converted_count = 0
    error_count = 0
    
    print(f"Found {len(png_files)} PNG files to convert...")
    print(f"Settings: Quality={quality}, Lossless={lossless}, Recursive={recursive}")
    print("-" * 50)
    
    for png_file in png_files:
        try:
            # Calculate relative path from input folder
            relative_path = png_file.relative_to(input_path)
            
            # Create corresponding output path
            webp_file = output_path / relative_path.with_suffix('.webp')
            
            # Create output directory if it doesn't exist (for recursive mode)
            webp_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open and convert the image
            with Image.open(png_file) as img:
                # Convert RGBA to RGB if necessary for lossy compression
                if not lossless and img.mode in ('RGBA', 'LA'):
                    # Create a white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    else:
                        background.paste(img)
                    img = background
                
                # Save as WebP
                save_kwargs = {'format': 'WebP'}
                if lossless:
                    save_kwargs['lossless'] = True
                else:
                    save_kwargs['quality'] = quality
                    save_kwargs['method'] = 6  # Best compression method
                
                img.save(webp_file, **save_kwargs)
            
            # Get file sizes for comparison
            original_size = png_file.stat().st_size
            new_size = webp_file.stat().st_size
            compression_ratio = (1 - new_size / original_size) * 100
            
            print(f"✓ {relative_path} -> {webp_file.name}")
            print(f"  Size: {original_size:,} bytes -> {new_size:,} bytes ({compression_ratio:.1f}% reduction)")
            
            converted_count += 1
            
        except Exception as e:
            print(f"✗ Error converting {png_file.relative_to(input_path)}: {str(e)}")
            error_count += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"Successfully converted: {converted_count} files")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files")

def main():
    parser = argparse.ArgumentParser(description='Convert PNG images to WebP format')
    parser.add_argument('folder', default='gen_out_sr_4096', help='Path to folder containing PNG images')
    parser.add_argument('-o', '--output', help='Output folder (default: input_folder + "_webp")')
    parser.add_argument('-q', '--quality', type=int, default=80, 
                       help='Compression quality (0-100, default: 80)')
    parser.add_argument('-l', '--lossless', action='store_true',
                       help='Use lossless compression')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Process subfolders recursively')
    parser.add_argument('--remove-original', action='store_true',
                       help='Remove original PNG files after conversion')
    
    args = parser.parse_args()
    
    # Validate quality parameter
    if not 0 <= args.quality <= 100:
        print("Error: Quality must be between 0 and 100")
        sys.exit(1)
    
    # Convert images
    convert_png_to_webp(args.folder, args.output, args.quality, args.lossless, args.recursive)
    
    # Remove original files if requested
    if args.remove_original:
        input_path = Path(args.folder)
        pattern = "**/*.png" if args.recursive else "*.png"
        png_files = list(input_path.glob(pattern))
        
        print(f"\nRemoving {len(png_files)} original PNG files...")
        for png_file in png_files:
            try:
                png_file.unlink()
                print(f"✓ Removed {png_file.name}")
            except Exception as e:
                print(f"✗ Error removing {png_file.name}: {str(e)}")

if __name__ == "__main__":
    main()