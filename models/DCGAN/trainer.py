# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 22:37:18 2018

@author: jihunlee326
"""

# Pytorch, torchvision, numpy AND utils, network
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from utils import *
import network

cuda = True if torch.cuda.is_available() else False
print(cuda)

os.makedirs('../images', exist_ok=True)

# Hyper parameters
latent_dim = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
nc = 3
batch_size = 64
n_epochs = 200
sample_interval = 10
img_size = 64

# Configure data loader
transforms_ = [ transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ]
dataloader = DataLoader(ImageDataset("../datasets/CelebA_128crop_FD/CelebA/128_crop/", 
                                     transforms_=transforms_), batch_size=batch_size, shuffle=True)

# GAN Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = network.Generator(latent_dim=latent_dim)
discriminator = network.Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# gpu or cpu
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

##################
#    Training    #      
##################

for epoch in range(n_epochs):
    for i, imgs in enumerate(dataloader):
        
        # Configure real images and ground truths
        real_imgs = Variable(imgs.type(Tensor))
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        
        ##############################
        #    Train Discriminator     #      
        ##############################
        optimizer_D.zero_grad()
        
        # Configure nosie vector
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        
        # Generate a batch of images
        gen_imgs = generator(z)
        
        # Measure discriminator's abillity to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        
        ##########################
        #    Train Generator     #      
        ##########################
        optimizer_G.zero_grad()

        # Loss measures generator's abillity to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        #----------------------------#
        #    save model pre epoch    #    
        #----------------------------#
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
             % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25], '../images/%d.png' % batches_done, nrow=5, normalize=True)
