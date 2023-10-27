from torchvision.models import vit_h_14, ViT_H_14_Weights, resnet50, ResNet50_Weights
from torchvision.io import read_image
from PIL import Image
import os, argparse
import torch
import pandas as pd
import numpy as np
import pdb


def compute_acc(folder, names, target_name, weights, save_path, topk=3):

    print(f"Computing {folder}...")

    batch_size = 250
    
    scores = {}
    categories = {}
    indexes = {}
    for k in range(1,topk+1):
        scores[f'top{k}']= []
        indexes[f'top{k}']=[]
        categories[f'top{k}']=[]

    preprocess = weights.transforms()

    images = []
    for name in names:
        img = Image.open(os.path.join(folder,name))
        batch = preprocess(img)
        images.append(batch)

    if batch_size == None:
        batch_size = len(names)
    if batch_size > len(names):
        batch_size = len(names)
    images = torch.stack(images)

    for i in range(((len(names)-1)//batch_size)+1):
        batch = images[i*batch_size: min(len(names), (i+1)*batch_size)].to(device)
        with torch.no_grad():
            prediction = model(batch).softmax(1)
        probs, class_ids = torch.topk(prediction, topk, dim = 1)

        for k in range(1,topk+1):
            scores[f'top{k}'].extend(probs[:,k-1].detach().cpu().numpy())
            indexes[f'top{k}'].extend(class_ids[:,k-1].detach().cpu().numpy())
            categories[f'top{k}'].extend([weights.meta["categories"][idx] for idx in class_ids[:,k-1].detach().cpu().numpy()])

    case_numbers = []
    for i, name in enumerate(names):
        case_number = name.split('/')[-1].split('_')[1].replace('.png','').replace('.jpg','')
        case_numbers.append(int(case_number))

    dict_final = {'case_number': case_numbers}

    for k in range(1,topk+1):
        dict_final[f'category_top{k}'] = categories[f'top{k}'] 
        dict_final[f'index_top{k}'] = indexes[f'top{k}'] 
        dict_final[f'scores_top{k}'] = scores[f'top{k}'] 

    df_results = pd.DataFrame(dict_final)
    df_results.to_csv(save_path)


    average_acc = np.sum([dict_final[f"category_top{k}"].count(target_name) for k in range(1, topk+1)]) / len(names)


    return average_acc
