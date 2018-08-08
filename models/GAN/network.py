# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 09:50:14 2018

@author: jihunlee326
"""

import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Linear(128, 256),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Linear(256, 512),
                nn.BatchNorm1d(512, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh()
            )
    
    def forward(self, z, img_shape):
        img = self.model(z)
        out = img.view(img.size(0), *img_shape)
        
        return out


class Discriminator(nn.Module):
    # initializers
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        
    def forward(self, img):
        img = img.view(img.size(0), -1)
        out = self.model(img)
        
        return out