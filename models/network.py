# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:51:31 2018

@author: jihun326
"""
import utils
import torch.nn as nn


class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv_block = nn.Sequential(
                # [ConvTranspose2d] 
                # H_out = (H_in - 1) x stride - 2 x padding + kernel_size + output_padding
                # [-1, input_dim, 1, 1] -> [-1, 512, 4, 4]
                nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(512), 
                nn.ReLU(True),
                
                # [-1, 512, 4, 4] -> [-1, 256, 8, 8]
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                
                # [-1, 256, 8, 8] -> [-1, 128, 16, 16]
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                
                # [-1, 128, 16, 16] -> [-1, 64, 32, 32]
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                
                # [-1, 64, 32, 32] -> [-1, channels, 64, 64]
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        
        utils.weights_init_normal(self)
        
    def forward(self, z):
        # [-1, z] -> [-1, z, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.conv_block(z)
        
        return out
        
class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_block = nn.Sequential(
                # [Conv2d]
                # H_out = (H_in + 2 x padding - dilation x (kernel_size -1) -1) / strid + 1
                # [-1, 3, 64, 64] -> [-1, 64, 32, 32]
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                
                # [-1, 64, 32, 32] -> [-1, 128, 16, 16]
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                
                # [-1, 128, 16, 16] -> [-1, 256, 8, 8]
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                
                # [-1, 256, 8, 8] -> [-1, 512, 4, 4]
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                
                # [-1, 512, 4, 4] -> [-1, 1024, 1, 1]
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
        
        utils.weights_init_normal(self)
        
    def forward(self, img):
        # img = [-1, 3, 64, 64]
        out = self.conv_block(img)
        out = out.view(out.shape[0], -1)
        
        return out
