import argparse
import wandb
import os

import torch
import numpy as np
import pandas as pd
import csv

import pdb


import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from torchvision import transforms
import torchvision


def parse_args():
    parser = argparse.ArgumentParser(description="Script of text to image generation.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4", required=False)
    parser.add_argument("--delta_ckpt", type=str, default=None, help="path to model ckpt")
    parser.add_argument("--full_delta_ckpt", type=str, default=None, help="path to model ckpt")
    parser.add_argument("--defense_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=str, default=42, required=False)
    parser.add_argument("--prompt", type=str, default="data/prompts/imagenet.csv", help="Either prompts or path to prompt file")
    parser.add_argument("--num_images", type=int, default=1, required=False)
    parser.add_argument("--use_prompt_file", default=False, action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)



    args = parser.parse_args()
    return args


def main():

    args = parse_args()


    weight_dtype = torch.float32
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    image_save_transform = transforms.Compose([transforms.ToTensor()])



    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    if args.delta_ckpt is not None:
        print("restroting from delta model from previous version")
        unet.load_state_dict(torch.load(args.delta_ckpt))
    if args.full_delta_ckpt is not None:
        print("restroting from delta model from previous version")
        unet= torch.load(args.full_delta_ckpt)
    if args.defense_ckpt is not None:
        print("restroting from trained defense model")
        model_dict = unet.state_dict()
        model_dict.update(torch.load(args.defense_ckpt))
        unet.load_state_dict(model_dict)

    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unet,
                        revision=None,
                        torch_dtype=weight_dtype,
                    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    
    if args.prompt.endswith('.csv'):
        df = pd.read_csv(args.prompt)
        for idx, row in df.iterrows():
            seed = row["evaluation_seed"]
            guidance_scale = row["evaluation_guidance"]
            prompt = row["prompt"]
            case_num = row["case_number"]
            generator = generator.manual_seed(seed)
            print(f"Generating with prompt: {prompt}")
            for idx in range(args.num_images):
                if args.num_images > 1:
                    case_num = idx
                image = pipeline(prompt, num_inference_steps=30, guidance_scale=guidance_scale, generator=generator).images[0]
                torchvision.utils.save_image(image_save_transform(image), f"{args.output_dir}/{case_num}.png")
    else:
        images = []
        for idx in range(args.num_images):
            image = pipeline(args.prompt, num_inference_steps=30, generator=generator).images[0]
            torchvision.utils.save_image(image_save_transform(image), f"{args.output_dir}/{idx}.png")


if __name__=='__main__':
    main()