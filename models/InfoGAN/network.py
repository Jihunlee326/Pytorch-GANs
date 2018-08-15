# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:24:30 2018

@author: Administrator
"""
import utils
import torch.nn as nn

class Generator(nn.Module):
    # initializers
    def __init__(self, latent_dim, categorical_dim, continuous_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.continuous_dim = continuous_dim

        input_dim = latent_dim + categorical_dim + continuous_dim
        
        self.conv_block = nn.Sequential(
                # [ConvTranspose2d] 
                # H_out = (H_in - 1) x stride - 2 x padding + kernel_size + output_padding
                # [-1, input_dim, 1, 1] -> [-1, 512, 4, 4]
                nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=1, padding=0),
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
        
    def forward(self, input):
        # [-1, input(z+labels+code)] -> [-1 , input(z+labels+code), 1, 1]
        x = input.view(input.size(0), -1, 1, 1)
        out = self.conv_block(x)
        
        return out
    
class Discriminator(nn.Module):
    # initializers
    def __init__(self, categorical_dim):
        super(Discriminator, self).__init__()
        self.categorical_dim=categorical_dim
        
        self.conv_block = nn.Sequential(
                # [Conv2d]
                # H_out = (H_in + 2 x padding - dilation x (kernel_size -1) -1) / strid + 1
                # [-1, channels, 64, 64] -> [-1, 64, 32, 32]
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                
                # [-1, 64, 32, 32] -> [-1, 128, 16, 16]
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                
                # [-1, 128, 16, 16] -> [-1, 256, 8, 8]
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                
                # [-1, 256, 8, 8] -> [-1, 512, 4, 4]
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                
                # [-1, 512, 4, 4] -> [-1, 64, 1, 1] 
                # 갑자기 이렇게 체널이 줄어도 괜찮을까?
                nn.Conv2d(512, 64, kernel_size=4, stride=1, padding=0),
            )
        
        self.adv_layer = nn.Sequential( nn.Linear(64, 1),
                                        nn.Sigmoid())
        
        self.aux_layer = nn.Sequential( nn.Linear(64, categorical_dim),
                                        nn.Softmax())
        
        self.latent_layer = nn.Sequential( nn.Linear(64, 1))
        
        utils.weights_init_normal(self)   
    
    def forward(self, input):
        # [-1, channels, 64, 64]
        out = self.conv_block(input)
        out = out.view(out.shape[0], -1)
        # Prediction [-1, 1]
        validity = self.adv_layer(out)
        # Class [-1, categorical_dim]
        label = self.aux_layer(out)
        # CC_dim [-1, 1]
        latent_code = self.latent_layer(out)

        return validity, label, latent_code