# Hairy Dawg Text-to-Image Model (LoRA)
![t2i](dataset/Picture1.png)


This repository contains LoRA fine-tuned weights for generating images of **Hairy Dawg**, the University of Georgia's beloved mascot, using **Stable Diffusion 1.5** (or **SDXL**). The repo includes an example **Jupyter Notebook** to demonstrate how to apply the LoRA weights for image generation.

## Features
- Fine-tuned LoRA weights for Hairy Dawg text-to-image generation.
- Example notebook for inference using Stable Diffusion and LoRA.
- Training dataset images and captions for reference.
- Easy integration with Hugging Face Diffusers library.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install diffusers transformers accelerate safetensors
```

## How to Use

1. **Clone the Stable Diffusion model from Hugging Face:**
   ```python
   from diffusers import StableDiffusionPipeline
   from safetensors.torch import load_file

   base_model = "runwayml/stable-diffusion-v1-5"  # Change to SDXL if needed
   pipe = StableDiffusionPipeline.from_pretrained(base_model)
   ```

2. **Load LoRA weights:**
   ```python
   lora_weights = "path_to_lora_weights.safetensors"
   pipe.unet.load_attn_procs(load_file(lora_weights))
   ```

3. **Generate images:**
   ```python
   prompt = "Hairy Dawg standing in a football stadium, highly detailed, 4K"
   image = pipe(prompt).images[0]
   image.show()
   ```

## Dataset

The training dataset consists of **60 curated images of Hairy Dawg**, along with corresponding captions, used to fine-tune the model using **DreamBooth**.

## Training

The model was fine-tuned using **LoRA (Low-Rank Adaptation)** to efficiently adapt Stable Diffusion to Hairy Dawg's visual style. Training was performed on a **single A100 GPU**.

## Acknowledgments

- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) by Runway & CompVis.
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/main/en/index) for inference and training support.

## License

This project is shared for research and educational purposes. The LoRA weights should not be used for commercial applications without proper authorization.
