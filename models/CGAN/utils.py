# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:55:34 2018

@author: jihunlee326

"""
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# for use celebA(128x128)
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transforms = transforms.Compose(transforms_)
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.image_paths)
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
