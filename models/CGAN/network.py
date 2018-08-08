# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:30:26 2018

@author: jihunlee326
"""

import utils
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim, classes, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.classes = classes
        
        # image_size = 64
        img_shape = (channels, 64, 64)
        
        self.nn_block = nn.Sequential(
                # [ConvTranspose2d] 
                # H_out = (H_in - 1) x stride - 2 x padding + kernel_size + output_padding
                # [-1, input_dim, 1, 1] -> [-1, 512, 4, 4]
                nn.Linear(latent_dim+classes, 128),
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
        
        utils.weights_init_normal(self)
        
    def forward(self, input):
        # [-1, input(z+labels)] -> [-1 , input(z+labels), 1, 1]
        #x = input.view(input.size(0), -1, 1, 1)
        x = input.view(input.size(0), -1)
        out = self.nn_block(x)
        
        return out
    
class Discriminator(nn.Module):   
    # initializers
    def __init__(self, classes, channels):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.classes = classes
        
        # image_size = 64
        img_shape = (channels, 64, 64)
        
        self.model = nn.Sequential(
                nn.Linear(classes+int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1),
            )
        
    def forward(self, img):
        # [-1, input(z+labels)] -> [-1 , input(z+labels), 1, 1]
        #x = img.view(img.size(0), -1, 1, 1)
        x = img.view(img.size(0), -1)
        out = self.model(x)
        
        return out