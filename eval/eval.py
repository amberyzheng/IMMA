import argparse
import os
import pandas as pd
import numpy as np

from random import shuffle

import clip
import lpips
import torch
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import pathlib

import pdb

class Evaluator(object):
    def __init__(self, device, metric="clip") -> None:
        self.device = device
        self.model, clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.metric = metric
        if self.metric == "dino":
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        elif self.metric == "lpips":
            self.model = lpips.LPIPS(net='alex')

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor


    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images) if not self.metric == "dino" else self.model(images)


    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        if self.metric == "lpips":
            return 1.0 - torch.mean(self.model(src_images, generated_images))
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

imsize = 64
loader = transforms.Compose([
    transforms.Resize((imsize, imsize), antialias=True),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name, lpips=False):
    image = Image.open(image_name)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    if lpips:
        image = (image-0.5)*2
    return image.to(torch.float)

def get_files(path, epoch, is_lpips=False):
    file_names = sorted([file for file in path.glob(r'{}_*.png'.format(epoch).zfill(11))])
    return [image_loader(file, is_lpips) for file in file_names]

def get_reference_files(path):
    file_names = []
    for extension in ["jpg", "png", "jpeg"]:
        file_names.extend(list(path.glob(r'*.{}'.format(extension))))
    return [image_loader(file) for file in file_names]


if __name__=='__main__':
    parser = argparse.ArgumentParser(prog = 'eval', description = 'Evaluate IMMA')
    parser.add_argument('--reference_dir', help='dir of reference images', type=str, required=True)
    parser.add_argument('--base_dir', help='dir of images without IMMA', type=str, required=True)
    parser.add_argument('--imma_dir', help='dir of images with IMMA', type=str, required=True)
    parser.add_argument('--save_dir', help='path to save results', type=str, required=False, default="results")
    parser.add_argument('--metric', help='evaluation metric', type=str, required=False, default="clip", choices=['clip', 'dino', 'lpips'])


    args = parser.parse_args()
    metric = args.metric

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric_model = Evaluator(device=device, metric=metric)

    reference_dir = args.reference_dir
    imma_dir = args.imma_dir
    base_dir = args.base_dir

    path_reference = pathlib.Path(reference_dir)
    path_imma = pathlib.Path(imma_dir)
    path_base = pathlib.Path(base_dir)

    image_names = os.listdir(imma_dir)
    epoch_ids = sorted(set(name.split("_")[0] for name in image_names if '.png' in name or '.jpg' in name))

    files_reference = torch.cat(get_reference_files(path_reference), dim=0)

    df = pd.DataFrame()
    baseline_values = {}
    imma_values = {}
    for epoch_id in epoch_ids:
        files_imma = torch.cat(get_files(path_imma, epoch_id), dim=0)
        files_baseline = torch.cat(get_files(path_base, epoch_id), dim=0)
        if metric == "lpips":
            n_ref = len(list(files_reference))
            n_imma = len(list(files_imma))
            assert n_ref == n_imma
        # Compute two scores: (ref, base), (ref, imma)
        baseline_values[epoch_id] = metric_model.img_to_img_similarity(files_reference, files_baseline).detach().cpu().numpy() # compute metric for base
        imma_values[epoch_id] = metric_model.img_to_img_similarity(files_reference, files_imma).detach().cpu().numpy() # compute metric for imma

    df[f"w/o IMMA"] = pd.Series(baseline_values)
    df[f"w/ IMMA"] = pd.Series(imma_values)
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f"{args.save_dir}/{metric}.csv")

    
