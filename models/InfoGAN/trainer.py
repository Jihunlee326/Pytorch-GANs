# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 13:34:28 2018

@author: jihunlee326
"""

# Pytorch, torchvision, numpy AND utils, network
import os
import numpy as np
import itertools

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
code_dim = 1            # latent code
n_classes = 10          # number os classes for dataset (label)
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_channels = 3
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

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# InfoGAN Function (Multi-Nomial)
def gen_dc(batch_size, dim):
    codes=[]
    code = np.zeros((batch_size, dim))
    random_cate = np.random.randint(0, dim, batch_size)
    code[range(batch_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)

def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.

    return Variable(FloatTensor(y_cat))

# GAN Loss function
adversarial_loss = nn.MSELoss()
categorical_loss = nn.CrossEntropyLoss()
continuous_loss = nn.MSELoss()

# Initialize generator and discriminator
generator = network.Generator(latent_dim=latent_dim, categorical_dim=n_classes, continuous_dim=code_dim)
discriminator = network.Discriminator(categorical_dim=n_classes)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
# itertools.chain : iterable 객체를 연결, ex = itertools.chain([1, 2, 3], {'a', 'b', 'c'}) => ex = (1, 2, 3, 'a', 'b', 'c')
optimizer_info = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr, betas=(b1, b2))

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
        labels = to_categorical(labels.numpy(), num_columns=n_classes)
        valid = Variable(FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        
        # Configure noise vector, fake labels and ground truths
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = to_categorical(np.random.randint(0, n_classes, batch_size), num_columns=n_classes)
        gen_codes = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, code_dim))))
        fake = Variable(FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        
        # concatenate noise vector, fake labels and code_dimentaion
        concat_z = torch.cat((z, gen_labels, gen_codes), -1)
        
        #############################
        #    Train Discriminator    #      
        #############################
        discriminator.zero_grad()
        
        fake_imgs = generator(concat_z)
        
        # output of D for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)
        
        # output of D for fake images
        fake_pred, _, _ = discriminator(fake_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)
        
        # get the losses for discriminator
        discriminator_loss = (d_real_loss + d_fake_loss) * 0.5
        
        discriminator_loss.backward()
        optimizer_D.step
        
        
        #########################
        #    Train Generator    #      
        #########################
        generator.zero_grad()
        
        # output of D for fake images
        validity, _, _ = discriminator(fake_imgs)
        
        # get loss for generator
        generator_loss = adversarial_loss(validity, valid)
        
        # update generator
        generator_loss.backward()
        optimizer_G.step
        
        
        ################################
        #    Train Information Loss    #      
        ################################
        optimizer_info.zero_grad()
        
        
        sampled_labels = np.random.randint(0, n_classes, batch_size)
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)
        
        
        # Configure noise vector, fake labels and ground truths
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=n_classes)
        code_input = Variable(FloatTensor(np.random.normal(-1, 1, (batch_size, code_dim))))
        
        # concatenate noise vector, fake labels and code_dimentaion
        concat_z = torch.cat((z, label_input, code_input), -1)
        
        gen_imgs = generator(concat_z)
        _, pred_label, pred_code = discriminator(gen_imgs)
        
        info_loss = categorical_loss(pred_label, gt_labels) + continuous_loss(pred_code, code_input) * 0.5
        
        info_loss.backward()
        optimizer_info.step()
        
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                            discriminator_loss.item(), generator_loss.item(), info_loss.item()))

