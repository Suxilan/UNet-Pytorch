# coding:utf-8
#anaconda/anaconda3/envs/python37
'''
@File    :   transform.py
@Time    :   2023/08/08 13:23:11
@Author  :   Asuka 
@Contact :   shixulei@whu.edu.cn
@Desc    :   None
'''

import torch
import numpy as np
from torchvision import transforms
import random
from PIL import Image
class SegmentationTransforms:
    def __init__(self, mean, std):
        self.img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30, interpolation = Image.Resampling.BILINEAR),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.label_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30, interpolation=Image.Resampling.BILINEAR),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)  # Custom Lambda to keep label values as they are
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0)  # Custom Lambda to keep label values as they are
        ])

        self.seed = random.randint(0, 2**32 - 1)
        
    def __call__(self, sample):
        image, label = sample
        torch.manual_seed(self.seed)
        image = self.img_transform(image)
        torch.manual_seed(self.seed)
        label = self.label_transform(label)
        return image, label

    def valid_transform(self, sample):
        image, label = sample
        image = self.normalize(image)
        label = self.totensor(label)
        return image, label
