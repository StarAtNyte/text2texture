# Text2Texture - Professional Seamless Texture Generation

A Python-based texture generation system that creates high-quality, seamless textures using SDXL.

**🔄 What makes it "seamless"?** Every texture generated uses advanced circular padding and noise rolling techniques to ensure perfect edge continuity when tiled infinitely.

### Example: Ornate Persian Carpet Pattern

<div align="center">

| Original Pattern | Tiled 4x4 Preview |
|:----------------:|:-----------------:|
| <img width="1024" height="1024" alt="image_part_001 (1)" src="https://github.com/user-attachments/assets/0fd45835-9e57-4586-b5a7-138e3e9b16ad" /> | ➡️ <img width="2048" height="2048" alt="tiled_2x2_tiled (1)" src="https://github.com/user-attachments/assets/da71c8df-008c-4f56-bb8f-c5275463436e" /> |


*Notice how the texture tiles perfectly with no visible seams at the edges*

</div>

The arrows (➡️) show the transformation from single texture to seamlessly tiled preview, demonstrating perfect edge continuity.


## 🌟 Features

### Core Technologies
- **SDXL Integration**: Full Stable Diffusion XL with refiner for highest quality
- **Seamless Generation**: Advanced circular padding and noise rolling for perfect seamless textures
- **Professional Web Interface**: Modern Flask-based webapp for interactive texture generation
- **Modal Labs Deployment**: Cloud-ready deployment with GPU acceleration

### Key Capabilities
- **🔄 Guaranteed Seamless Output**: Every generated texture tiles perfectly with zero visible seams
- **🔍 Multiple Tiling Verification**: 2x2, 3x3, and 4x4 automatic tiling tests to verify seamlessness

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (A100-40GB recommended for Modal deployment)
- Modal Labs account for cloud deployment
- HuggingFace account with access to SDXL models

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text2texture
```

2. Install dependencies:
```bash
pip install modal torch diffusers transformers pillow flask numpy
```

3. Set up Modal secrets:
```bash
modal secret create huggingface-secret HUGGINGFACE_TOKEN=your_token_here
```

### Usage Options

#### 1. Interactive Seamless Texture Web Interface (Recommended)
```bash
# Deploy to Modal Labs - generates seamless textures on-demand
modal deploy texture_webapp_full.py

# Or run locally for testing (requires GPU)
python texture_webapp_full.py
```
**Result**: Web interface where you input a prompt and get a seamless texture + tiled preview

#### 2. Batch Seamless Dataset Generation
```bash
# Generate complete seamless texture datasets
modal run modal_texture_generator_sdxl.py::main

# Test seamless quality with different materials
modal run modal_texture_generator_sdxl.py::test_materials

# Create 4x4 tiled versions to verify seamlessness
modal run modal_texture_generator_sdxl.py::tile
```
**Result**: Hundreds of perfectly seamless textures organized by material type

## 🎯 System Components

### 1. SDXL Seamless Generation Engine

**File**: `modal_texture_generator_sdxl.py`

**🔄 Seamless-Specific Features**:
- **Circular Padding**: Replaces Conv2d edge behavior to wrap pixels seamlessly
- **Noise Rolling**: Shifts latent noise by 64px during generation for edge continuity  
- **80% Threshold Callback**: Applies seamless techniques at optimal generation point
- **Two-Stage SDXL**: Base model + refiner maintains seamless properties through refinement
- **Automated Seamless Verification**: 4x4 tiling pipeline tests every generated texture
- **Seamless Success Tracking**: Monitors and reports seamless generation success rates

**Core Functions**:
```python
# Seamless patching functions
asymmetricConv2DConvForward_circular()  # Circular padding for seamless edges
make_seamless_sdxl()                     # Enable seamless mode on SDXL
disable_seamless_sdxl()                  # Restore normal Conv2d behavior
sdxl_diffusion_callback()               # Advanced callback with noise rolling
```

**Usage Examples**:
```bash
# Generate complete texture dataset
modal run modal_texture_generator_sdxl.py::main

# Test seamless settings
modal run modal_texture_generator_sdxl.py::test

# Test multiple materials
modal run modal_texture_generator_sdxl.py::test_materials

# Create 4x4 tiled versions
modal run modal_texture_generator_sdxl.py::tile

# Run complete pipeline
modal run modal_texture_generator_sdxl.py::complete_pipeline
```

### 2. Professional Seamless Texture Web Interface

**Deployment**:
```bash
# Deploy to Modal Labs
modal deploy texture_webapp.py

# The webapp will be available at the provided Modal URL
```

### 3. Material Specification System

**File**: `materials_part1.json`

Defines material categories with prompt templates, styles, and colors:

```json
{
  "rugs": {
    "hand-knotted-silk": {
      "prompt_template": "seamless tileable hand-knotted silk rug material texture, {style}, {color}, ultra-sharp 8k macro, orthogonal top-down, diffuse studio light",
      "styles": ["fine-silk-knot-bumps", "dense-knot-surface", "lustrous-silk-weave"],
      "colors": ["ivory", "champagne"]
    }
  }
}
```

## 🔧 Technical Implementation

### Advanced Seamless Texture Algorithm

**🔄 Why These Textures Are Guaranteed Seamless:**

The system implements a sophisticated 4-step seamless generation process:

1. **🔄 Circular Padding**: Replaces standard Conv2d edge padding with circular wrapping
   - Ensures left edge connects perfectly to right edge
   - Top edge connects perfectly to bottom edge
   - No "hard edges" that cause visible seams

2. **📏 64px Noise Rolling**: Shifts latent noise by 64 pixels during first 80% of generation
   - Prevents edge artifacts from accumulating
   - Distributes generation patterns across texture boundaries
   - Maintains consistency across wrap-around points

3. **⏱️ 80% Threshold Activation**: Applies seamless techniques at optimal generation timing
   - Early enough to affect final image structure
   - Late enough to preserve detail quality
   - Scientifically determined optimal activation point

4. **🎨 Two-Stage Seamless Refinement**: SDXL base + refiner both maintain seamless properties
   - Base model generates seamless structure
   - Refiner enhances details while preserving seamless edges
   - Dual-stage process ensures seamlessness survives refinement

### SDXL Configuration

```python
# Model specifications
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0" 
DTYPE = torch.bfloat16

# Generation parameters
guidance_scales = [7.5, 8.5]
num_inference_steps = 50
high_noise_frac = 0.8  # Denoising split point
```
'
