"""
Created on Sun Jul 29 14:39:34 2018

@author: jihun.lee326@gmail.com
"""

# Pytorch, torchvision, numpy 그리고 datasets
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch
import torch.nn.functional as F

from PIL import Image

from datasets import *

# Hyper parameters
latent_dim = 128
cc_dim = 1
dc_dim = 10
img_size = 64
batch_size = 64
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
n_epochs = 200

cuda = True if torch.cuda.is_available() else False
print(cuda)

# Configure data loader
os.makedirs('../data/img_align_celeba/', exist_ok=True)

transforms_ = [ transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset('../data/img_align_celeba/', transforms_=transforms_), 
                       batch_size=12, shuffle=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, latent_dim, cc_dim=1, dc_dim=10):
        super(Generator, self).__init__()
        
        self.conv_block = nn.Sequential(
            # [-1 , z + cc + dc, 1, 1] -> [-1, 512, 4, 4]
            nn.ConvTranspose2d(latent_dim + cc_dim + dc_dim, 1024, 4, 1, 0, bias=False),
            
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # [-1, 256, 8, 8]
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # [-1, 128, 16, 16]
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # [-1, 3, 32, 32]
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        # [-1, z] -> [-1, z, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.conv_block(z)
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, cc_dim=1, dc_dim=10):
        super(Discriminator, self).__init__()
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim
        
        self.conv_block = nn.Sequential(
            # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
            nn.Conv2d(3, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [-1, 128, 7, 7]
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # [-1, 512, 4, 4]
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            # [-1, 1 + cc_dim + dc_dim, 1, 1]
            nn.Conv2d(1024, 1 + cc_dim + dc_dim, 4, stride=1, padding=0)
        )
        
    def forward(self, x):
        # [-1, 1 + cc_dim + dc_dim]
        out = self.conv_block(x).squeeze()
        out[:, 0] = F.sigmoid(out[:, 0].clone())
        out[:, self.cc_dim+1:self.cc_dim+1+self.dc_dim] = F.softmax(out[:, self.cc_dim+1:self.cc_dim+1+self.dc_dim].clone())
        return out
    
# Network
generator = Generator(latent_dim, cc_dim, dc_dim)
discriminator = Discriminator(cc_dim, dc_dim)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if cuda:
    generator.cuda()
    discriminator.cuda()

# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)

for epoch in range(n_epochs):
    for i, images in enumerate(dataloader):
        images = Variable(images.type(Tensor))
        
        batch_size = images.size(0)
        noise = Variable(torch.randn(batch_size, latent_dim))
        cc = Variable(torch.Tensor(np.random.randn(batch_size, cc_dim)))
        dc = Variable(gen_dc(batch_size, dc_dim))
