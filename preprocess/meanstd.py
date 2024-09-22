# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   meanstd.py
@Time    :   2023/08/07 18:14:20
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

def calculate(path):
    files = os.listdir(path)
    trans = transforms.ToTensor()
    num_channels = 0
    mean = torch.zeros(3) # Assuming 3 channels for RGB images (for initialization)
    std = torch.zeros(3)  # Assuming 3 channels for RGB images (for initialization)
 
    for file in tqdm(files):
        img = Image.open(os.path.join(path, file))
        img = trans(img)
        
        if num_channels == 0:
            num_channels = img.shape[0]
            mean = torch.zeros(num_channels)
            std = torch.zeros(num_channels)
        
        for i in range(num_channels):
            mean[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()
    
    mean.div_(len(files))
    std.div_(len(files))
    print("Mean:", mean)
    print("Standard Deviation:", std)
    return mean, std


image_path = "data/train/images/"
calculate(image_path)