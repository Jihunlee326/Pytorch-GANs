# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:32:56 2018

@author: jihunlee326
"""

# Pytorch, torchvision, numpy AND utils, network
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

import network

cuda = True if torch.cuda.is_available() else False
print(cuda)

os.makedirs('../images', exist_ok=True)

# Hyper parameters
latent_dim = 100        # dimension of noise-vector
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_channels = 3
n_classes = 10          # dimension of code-vector (label)
batch_size = 64
n_epochs = 200
sample_interval = 10
img_size = 64

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
# Information Loss fuction
auxiliary_loss = nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = network.Generator(latent_dim=latent_dim, channels=n_channels)
discriminator = network.Discriminator(channels=n_channels)

# Label embedding
label_emb = nn.Embedding(n_classes, latent_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()
    label_emb.cuda()
    
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# gpu or cpu
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


##################
#    Training    #      
##################

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        # Configure real images, labels and ground truths
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                
        # Configure noise vector, fake labels and ground truths
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        
        # concatenate noise vector and fake labels
        z.mul_(label_emb(gen_labels))
        
        
        ##############################
        #    Traini Discriminator    #      
        ##############################
        discriminator.zero_grad()
        
        fake_imgs = generator(z)
        
        # output of D for real images
        real_pred, real_label = discriminator(real_imgs)
        # output of D for fake images
        fake_pred, fake_label = discriminator(fake_imgs.detach())
        
        # get loss for discriminator
        real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_label, labels)) * 0.5
        fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_label, gen_labels)) * 0.5
        discriminator_loss = (real_loss + fake_loss) * 0.5
        
        # update discriminator
        discriminator_loss.backward()
        optimizer_D.step()
        
        
        ##########################
        #    Traini Generator    #      
        ##########################
        generator.zero_grad()
        
        # output of D for fake images
        validity, pred_label = discriminator(fake_imgs) # 변수명 수정할 것
        
        # get loss for generator
        generator_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)) * 0.5
        
        # update generator
        generator_loss.backward()
        optimizer_G.step
        
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