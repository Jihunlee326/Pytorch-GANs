# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:27:24 2018

@author: jihunlee326
"""

# Pytorch, torchvision, numpy AND utils, network
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

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

img_shape = (nc, img_size, img_size)


# Configure data loader
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../../datasets/cifar10/', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Resize((img_size, img_size), Image.BICUBIC),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])), batch_size=batch_size, shuffle=True)

# GAN Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = network.Generator(latent_dim=latent_dim, img_shape=img_shape)
discriminator = network.Discriminator(img_shape=img_shape)

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
    for i, (imgs, _) in enumerate(dataloader):
        
        # Configure real images and ground truths
        real_imgs = Variable(imgs.type(Tensor))
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        
        
        ##############################
        #    Traini Discriminator    #      
        ##############################        
        optimizer_D.zero_grad()
        
        # Configure noise vector
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        
        # Generate a fake images
        fake_imgs = generator(z, img_shape)
        
        # get loss for discriminator
        real_loss = (adversarial_loss(discriminator(real_imgs), valid))
        fake_loss = (adversarial_loss(discriminator(fake_imgs.detach()), fake))
        discriminator_loss = (real_loss + fake_loss) / 2
        
        # update discriminator
        discriminator_loss.backward()
        optimizer_D.step()
        
        
        ##########################
        #    Traini Generator    #      
        ##########################
        generator.zero_grad()
        
        # get loss for generator
        generator_loss = adversarial_loss(discriminator(fake_imgs), valid)
        
        # update generator
        generator_loss.backward()
        optimizer_G.step()
        
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" 
               % (epoch, n_epochs, i, len(dataloader), discriminator_loss.item(), generator_loss.item()))
        
        
        #-----------------------------
        #    save model pre epoch    #    
        #-----------------------------
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(fake_imgs.data[:25], '../images/%d.png' % batches_done, nrow=5, normalize=True)
            torch.save(discriminator, f'../chkpts/d_{epoch:03d}.pth')
            torch.save(generator, f'../chkpts/g_{epoch:03d}.pth')
