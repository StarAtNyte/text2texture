# Texture Generation

A comprehensive Python-based texture generation system that creates high-quality, seamless textures using AI models. The system combines Stable Diffusion (SDXL) for texture generation with SwinIR for 4x super-resolution upscaling.

## Features

- **Multiple AI Models**: Support for both Stable Diffusion 3.5 Medium and SDXL with LoRA
- **Comprehensive Material Library**: 7 categories with 20+ materials including rugs, fabrics, papers, leather, wood, and metal
- **Super-Resolution**: Automatic 4x upscaling using SwinIR (1024×1024 → 4096×4096)
- **Smart Resumption**: Skip already processed images for interrupted runs
- **Memory Optimization**: RTX 4090 optimized with batch processing and VRAM management
- **Format Conversion**: PNG to WebP conversion for web-friendly formats

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (RTX 4090 recommended)
- 32GB+ RAM for large batch processing

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text2texture
```

2. Install dependencies:
```bash
pip install torch torchvision diffusers transformers pillow python-dotenv huggingface-hub
pip install basicsr timm opencv-python  # For SwinIR
```

3. Set up environment variables:
```bash
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### Basic Usage

#### Generate with Stable Diffusion 3.5 Medium
```bash
python generate_many_textures.py
```

#### Generate with SDXL + LoRA (Recommended for textures)
```bash
python generate_textures_sdxl.py
```

#### Super-resolve existing images
```bash
python superresolve.py input_folder output_folder --batch_size 3 --patch_wise
```

#### Convert to WebP
```bash
python pngtowebp.py input_folder output_folder
```

## System Architecture

### Core Components

1. **Texture Generation**
   - `generate_many_textures.py`: SD3.5 Medium pipeline
   - `generate_textures_sdxl.py`: SDXL + LoRA pipeline (recommended)
   - `app.py`: Interactive single texture generation

2. **Super-Resolution**
   - `superresolve.py`: Batch SwinIR processing
   - `standalone_superresolve.py`: Standalone version

3. **Utilities**
   - `pngtowebp.py`: Format conversion
   - `generation_plan.md`: Material specifications

### Directory Structure

```
text2texture/
├── README.md
├── CLAUDE.md                    # Development instructions
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── generate_many_textures.py    # SD3.5 batch generation
├── generate_textures_sdxl.py    # SDXL batch generation
├── superresolve.py             # Batch super-resolution
├── pngtowebp.py                # Format conversion
├── app.py                      # Interactive generation
├── generation_plan.md          # Material specifications
├── gen_out_sd_1024/           # Generated 1024×1024 images
├── gen_out_sr_4096/           # Super-resolved 4096×4096 images
└── gen_out_sr_4096_webp/      # WebP converted images
```

## Material Categories

### Rugs
- **Wool-pile**: cut-pile, loop-pile, saxony, frieze
- **Jute-boucle**: chunky, fine, mixed-yarn, herringbone
- **Sisal-flatweave**: basket, herringbone, chevron, plain
- **Persian-traditional**: tabriz, kashan, isfahan, heriz

### Fabrics
- **Cotton-canvas**: plain-weave, twill, duck-canvas, heavy-weight
- **Linen-textile**: fine-weave, loose-weave, slub-texture, stonewashed
- **Denim-fabric**: raw-selvage, washed, stretch, heavyweight
- **Silk-textile**: charmeuse, taffeta, dupioni, chiffon

### Papers
- **Watercolor-paper**: hot-press, cold-press, rough, medium
- **Kraft-paper**: smooth, textured, recycled, heavy-weight
- **Handmade-paper**: mulberry, bamboo, cotton-rag, hemp
- **Parchment-paper**: aged, smooth, textured, antique

### Leather
- **Full-grain-leather**: smooth, pebbled, pull-up, distressed
- **Suede-leather**: nubuck, brushed, soft-suede, microsuede

### Wood
- **Hardwood-grain**: oak, maple, walnut, cherry
- **Reclaimed-wood**: barn-wood, driftwood, weathered, rustic

### Metal
- **Brushed-metal**: aluminum, steel, brass, copper
- **Oxidized-metal**: patina, rust, verdigris, aged

## Configuration

### Environment Variables

```bash
# Required for gated models
HUGGINGFACE_TOKEN=your_token_here
```

### Common Parameters

```python
# Generation settings
WIDTH, HEIGHT = 1024, 1024
STEPS = 28
GUIDANCE_SCALES = [6.5, 7.5]  # Reduced for faster generation

# Super-resolution settings
BATCH_SIZE = 3                 # Reduce if OOM
PATCH_WISE = True             # Enable for lower VRAM
DELAY_BETWEEN_BATCHES = 2.0   # Seconds between batches
```

## Performance Optimization

### Memory Management
- Sequential processing (SD → offload → super-resolution)
- Batch processing with configurable sizes
- Patch-wise processing for super-resolution
- Automatic CUDA cache clearing

### Hardware Recommendations
- **GPU**: RTX 4090 (24GB VRAM) recommended
- **RAM**: 32GB+ for large batch processing
- **Storage**: Fast SSD for model storage and processing

### Reducing Memory Usage
```bash
# Smaller batch size
--batch_size 1

# Enable patch-wise processing
--patch_wise

# Increase delay between batches
--delay_between_batches 5.0
```

## File Naming Convention

### Generated Images
```
{material}_{style}_{color}_cfg{guidance}_seed{seed}.png
```

### Super-resolved Images
```
{original_name}_swinir_x4.png
```

### Examples
- `wool-pile_cut-pile_charcoal_cfg7.5_seed1234567890.png`
- `wool-pile_cut-pile_charcoal_cfg7.5_seed1234567890_swinir_x4.png`

## Advanced Usage

### Custom Material Generation

Edit the `TEXTURE_CATEGORIES` dictionary in `generate_many_textures.py`:

```python
TEXTURE_CATEGORIES = {
    "custom_category": {
        "custom_material": {
            "prompt_template": "colormap, seamless tileable {material}, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases",
            "styles": ["style1", "style2", "style3"],
            "colors": ["color1", "color2", "color3"]
        }
    }
}
```

### Batch Processing with Custom Parameters

```bash
# High-quality generation with more steps
python generate_many_textures.py --steps 50 --guidance_scales 7.0,8.0

# Super-resolution with custom settings
python superresolve.py input/ output/ --batch_size 2 --tile_size 480 --delay_between_batches 3.0
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 1

# Enable patch-wise processing
--patch_wise

# Reduce image dimensions (if needed)
WIDTH, HEIGHT = 768, 768
```

**Model Download Failures**
```bash
# Check HuggingFace token
echo $HUGGINGFACE_TOKEN

# Manually download models
huggingface-cli download stabilityai/stable-diffusion-3.5-medium
```

**Slow Generation**
```bash
# Use SDXL instead of SD3.5
python generate_textures_sdxl.py

# Reduce guidance scale count
CFG_VALUES = [7.0]  # Single value instead of multiple
```

### Performance Tips

1. **Use SDXL with LoRA** for better texture quality
2. **Enable patch-wise processing** for lower VRAM usage
3. **Adjust batch sizes** based on your GPU memory
4. **Monitor system resources** during generation

## Model Information

### Stable Diffusion Models
- **SD3.5 Medium**: `stabilityai/stable-diffusion-3.5-medium`
- **SDXL Base**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Texture LoRA**: `dog-god/texture-synthesis-sdxl-lora`

### Super-Resolution
- **SwinIR-L**: 4x upscaling optimized for real-world images
- **Model**: `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your materials to the texture categories
4. Test with small batches first
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Stability AI for Stable Diffusion models
- SwinIR team for super-resolution model
- Hugging Face for model hosting and diffusers library
- dog-god for texture synthesis LoRA

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review CLAUDE.md for development guidance
3. Open an issue on GitHub
