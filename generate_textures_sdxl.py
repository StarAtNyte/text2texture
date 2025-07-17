#!/usr/bin/env python3
"""
generate_many_textures.py
SDXL + texture-synthesis LoRA (or Flux-Dev – see comments)
Creates 41 × 5 × 5 × 5 = 5,125 rug textures → 1024² → 4× SwinIR → 4096²
RTX 4090 friendly (fp16, attention-slicing, 3-img batches)
"""
import os, sys, json, random, itertools, subprocess
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
# For Flux-Dev, swap to:
# from optimum.quanto import qfloat8
# from diffusers import FluxPipeline

# ---------- CONFIG ----------
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_ID = "dog-god/texture-synthesis-sdxl-lora"
LORA_WEIGHT_NAME = "texture-synthesis-topdown-base-condensed.safetensors"
LORA_SCALE = 0.8
WIDTH, HEIGHT = 1024, 1024
STEPS = 28
GUIDANCE_SCALES = [6.0, 6.5, 7.0, 7.5, 8.0]
SEED = None # random per image
NEGATIVE_PROMPT = ("blurry, lowres, watermark, text, signature, folds, "
                   "creases, perspective, 3d, cartoon, illustration")
OUT_DIR_SD = Path("gen_out_sd_1024")
OUT_DIR_SR = Path("gen_out_sr_4096")

# ---------- COMPLETE MATERIAL BANK ----------
# Each material has its own specific styles and colors from the generation plan
MATERIALS = {
    "wool-pile": {
        "styles": ["cut-pile", "loop-pile", "saxony", "frieze", "plush"],
        "colors": ["sand", "charcoal", "indigo", "sage", "terracotta"]
    },
    "wool-loop": {
        "styles": ["berber", "level-loop", "multi-level", "sisal-look", "textured"],
        "colors": ["cream", "slate", "navy", "olive", "rust"]
    },
    "cotton-dhurrie": {
        "styles": ["plain", "striped", "geometric", "diamond", "kilim"],
        "colors": ["white", "denim", "ochre", "teal", "black"]
    },
    "jute-boucle": {
        "styles": ["chunky", "fine", "mixed-yarn", "herringbone", "chevron"],
        "colors": ["natural", "walnut", "moss", "stone", "graphite"]
    },
    "nylon-saxony": {
        "styles": ["cut-pile", "trackless", "plush", "sculptured", "patterned"],
        "colors": ["silver", "taupe", "burgundy", "emerald", "ivory"]
    },
    "polyester-shag": {
        "styles": ["long-pile", "short-pile", "mixed-fiber", "metallic", "ombré"],
        "colors": ["blush", "aqua", "mauve", "gold", "onyx"]
    },
    "sisal-flatweave": {
        "styles": ["basket", "herringbone", "chevron", "plain", "bordered"],
        "colors": ["sand", "khaki", "charcoal", "indigo", "rust"]
    },
    "seagrass-basketweave": {
        "styles": ["tight", "loose", "diamond", "spiral", "checker"],
        "colors": ["natural", "sage", "driftwood", "slate", "pebble"]
    },
    "coir-herringbone": {
        "styles": ["fine", "chunky", "bordered", "printed", "geometric"],
        "colors": ["coconut", "espresso", "forest", "slate", "white"]
    },
    "merino-wool-plush": {
        "styles": ["dense-pile", "hand-knotted", "velvet-finish", "hand-tufted", "lustrous-sheen"],
        "colors": ["ivory", "dove-gray", "rosewater", "champagne", "sapphire"]
    },
    "lambswool-knit": {
        "styles": ["cable-knit", "ribbed", "garter-stitch", "chunky-braid", "moss-stitch"],
        "colors": ["cream", "heather-gray", "oatmeal", "blush-pink", "sky-blue"]
    },
    "alpaca-fur": {
        "styles": ["long-hair", "silky-plush", "natural-pelt", "surfaced-tip", "brushed-finish"],
        "colors": ["fawn", "white", "silver-gray", "vicuña-brown", "charcoal"]
    },
    "viscose-pile": {
        "styles": ["silky-sheen", "low-pile", "high-pile", "carved", "distressed"],
        "colors": ["pearl", "silver", "pewter", "celadon", "plum"]
    },
    "cotton-pile": {
        "styles": ["bath-mat", "printed", "tufted", "reversible", "looped"],
        "colors": ["optic-white", "butter-yellow", "mint-green", "coral", "navy"]
    },
    "polypropylene-berber": {
        "styles": ["level-loop", "multi-level-loop", "flecked", "solid", "patterned"],
        "colors": ["beige", "greige", "mocha", "granite", "steel-blue"]
    },
    "hemp-flatweave": {
        "styles": ["soumak", "plain-weave", "twill", "knotted", "slub-textured"],
        "colors": ["natural-hemp", "slate", "ochre", "forest-green", "brick"]
    },
    "chenille-texture": {
        "styles": ["plush", "ribbed", "vintage-wash", "jacquard", "low-profile"],
        "colors": ["ruby", "mustard", "peacock", "ash", "eggplant"]
    },
    "microfiber-shag": {
        "styles": ["ultra-soft", "dense-pile", "noodle-pile", "silky", "high-low"],
        "colors": ["cloud-white", "lilac", "charcoal", "turquoise", "fuchsia"]
    },
    "jute-and-wool-blend": {
        "styles": ["textured-stripe", "diamond-motif", "chunky-loop", "flatweave", "heathered"],
        "colors": ["oatmeal", "ivory", "slate-gray", "denim-blue", "natural"]
    },
    "linen-and-cotton-blend": {
        "styles": ["herringbone", "striped", "stonewashed", "fringed", "block-print"],
        "colors": ["flax", "washed-black", "seafoam", "terracotta", "ecru"]
    }
}

# ---------- PIPELINE ----------
def build_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LORA_ID, weight_name=LORA_WEIGHT_NAME)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    return pipe

# ---------- GENERATE ----------
def run_sd(pipe):
    OUT_DIR_SD.mkdir(exist_ok=True, parents=True)
    
    # Generate all combinations
    tasks = []
    for material, specs in MATERIALS.items():
        for style in specs["styles"]:
            for color in specs["colors"]:
                for guidance in GUIDANCE_SCALES:
                    tasks.append((material, style, color, guidance))
    
    total = len(tasks)
    print(f"Generating {total} SDXL images ({len(MATERIALS)} materials × 5 styles × 5 colors × 5 guidance scales)...")
    
    for idx, (mat, sty, col, cfg) in enumerate(tasks, 1):
        # Create material-specific prompt based on the generation plan
        if mat == "cotton-dhurrie":
            base_prompt = f"colormap, seamless tileable {mat} flat-weave rug"
        elif mat == "seagrass-basketweave":
            base_prompt = f"colormap, seamless tileable seagrass-basketweave rug"
        elif mat == "coir-herringbone":
            base_prompt = f"colormap, seamless tileable {mat} rug"
        elif mat == "chenille-texture":
            base_prompt = f"colormap, seamless tileable chenille-texture rug"
        elif mat == "jute-and-wool-blend":
            base_prompt = f"colormap, seamless tileable jute-and-wool blend rug"
        elif mat == "linen-and-cotton-blend":
            base_prompt = f"colormap, seamless tileable linen-and-cotton blend rug"
        elif mat == "flocked-velvet":
            base_prompt = f"colormap, seamless tileable flocked-velvet carpet"
        elif mat == "olefin-outdoor":
            base_prompt = f"colormap, seamless tileable olefin-outdoor rug"
        elif mat == "felted-wool-ball":
            base_prompt = f"colormap, seamless tileable felted-wool-ball rug"
        elif mat == "acrylic-outdoor":
            base_prompt = f"colormap, seamless tileable acrylic-outdoor rug"
        elif mat == "cut-and-loop-pile":
            base_prompt = f"colormap, seamless tileable cut-and-loop pile carpet"
        elif mat == "wool-and-silk-blend":
            base_prompt = f"colormap, seamless tileable wool-and-silk blend rug"
        elif mat == "tencel-pile":
            base_prompt = f"colormap, seamless tileable tencel-pile rug"
        elif mat == "wool-and-nylon-blend":
            base_prompt = f"colormap, seamless tileable wool-and-nylon blend carpet"
        elif mat == "nylon-frieze":
            base_prompt = f"colormap, seamless tileable nylon-frieze carpet"
        elif mat == "polyester-loop":
            base_prompt = f"colormap, seamless tileable polyester-loop carpet"
        elif mat == "triexta-cut-pile":
            base_prompt = f"colormap, seamless tileable triexta-cut-pile carpet"
        elif mat == "leather-shag":
            base_prompt = f"colormap, seamless tileable leather-shag rug"
        elif mat == "hair-on-hide":
            base_prompt = f"colormap, seamless tileable hair-on-hide rug"
        elif mat == "bamboo-silk":
            base_prompt = f"colormap, seamless tileable bamboo-silk rug"
        elif mat == "recycled-PET-fiber":
            base_prompt = f"colormap, seamless tileable recycled-PET-fiber rug"
        elif mat == "braided-rag":
            base_prompt = f"colormap, seamless tileable braided-rag rug"
        elif mat == "mohair-plush":
            base_prompt = f"colormap, seamless tileable mohair-plush rug"
        elif mat == "cashmere-plush":
            base_prompt = f"colormap, seamless tileable cashmere-plush rug"
        elif mat == "metallic-lurex-thread":
            base_prompt = f"colormap, seamless tileable metallic-lurex-thread rug"
        elif mat == "abaca-fiber":
            base_prompt = f"colormap, seamless tileable abaca-fiber rug"
        else:
            # Default pattern for most materials
            base_prompt = f"colormap, seamless tileable {mat} rug"
        
        prompt = (f"{base_prompt}, {sty}, {col}, "
                 f"ultra-sharp 8k macro, orthogonal top-down, diffuse studio light, no perspective, no creases")
        
        seed = random.randint(0, 2**32-1)
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=cfg,
            generator=generator,
            cross_attention_kwargs={"scale": LORA_SCALE}
        ).images[0]
        fname = f"{mat}_{sty}_{col}_cfg{cfg}_seed{seed}.png"
        image.save(OUT_DIR_SD / fname)
        print(f"[{idx:>4}/{total}] {fname}")

# ---------- UPSCALE 4× ----------
def run_swinir():
    script = Path(__file__).with_name("superresolve.py") # your existing upscaler script
    cmd = [
        sys.executable, str(script),
        str(OUT_DIR_SD),
        str(OUT_DIR_SR),
        "--batch_size", "3",
        "--patch_wise",
        "--delay_between_batches", "2.0"
    ]
    subprocess.run(cmd, check=True)

# ---------- MAIN ----------
if __name__ == "__main__":
    pipe = build_pipe()
    run_sd(pipe)
    torch.cuda.empty_cache()
    run_swinir()
    print("All done! 4096² textures in", OUT_DIR_SR)