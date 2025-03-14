# Hairy Dawg Text-to-Image Model (LoRA)
![t2i](dataset/Picture1.png)

This repository is part of the [GenAI Competition 2.0](https://www.franklin.uga.edu/news/stories/2024/genai-competition-20), showcasing an AI-generated mascot model for the University of Georgia. It contains **LoRA fine-tuned weights** for generating images of **Hairy Dawg**, the Georgia Bulldogs' beloved mascot, using **Stable Diffusion 1.5** (or **SDXL**). The repo includes an example **Jupyter Notebook** to demonstrate how to apply the LoRA weights for image generation.

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
   You can use **Stable Diffusion 1.5** or **SDXL** as the base model. To clone SDXL, run:

   ```bash
   git lfs install
   git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
   ```

   Or, load the model in Python:

   ```python
   from diffusers import StableDiffusionPipeline
   from safetensors.torch import load_file

   base_model = "stabilityai/stable-diffusion-xl-base-1.0"  # Use SDXL
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

## Self-Training

If you want to fine-tune the model yourself, you can use the following **DreamBooth LoRA** training script:

```bash
accelerate launch /scratch/mk47369/1.Research/python/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py   --pretrained_model_name_or_path="./"   --instance_data_dir="./dataset/instance_images"   --class_data_dir="./dataset/class_images"   --output_dir="./dataset/dreambooth_output_2000"   --instance_prompt="a photo of Hairy Dawg, the Georgia Bulldogs mascot"   --class_prompt="a photo of a bulldog"   --resolution=1024   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=2e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --num_train_epochs=500   --checkpointing_steps=500   --enable_xformers_memory_efficient_attention   --mixed_precision="fp16"
```

## Dataset

The training dataset consists of **60 curated images of Hairy Dawg**, along with corresponding captions, used to fine-tune the model using **DreamBooth**.

## Training

The model was fine-tuned using **LoRA (Low-Rank Adaptation)** to efficiently adapt Stable Diffusion to Hairy Dawg's visual style. Training was performed on a **single A100 GPU**.

## Acknowledgments

- [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) by Stability AI.
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/main/en/index) for inference and training support.

## License

This project is shared for research and educational purposes. The LoRA weights should not be used for commercial applications without proper authorization.
