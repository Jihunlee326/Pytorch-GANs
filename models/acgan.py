# Auxiliary Classifier GANs
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

# Hyper-parameters
latent_dim = 100
ngf = 64
ndf = 64
nc = 3

lr = 0.0002
batch_size = 64
img_size = 64
nz = 100
num_classes = 10
n_epochs = 200

cuda = True if torch.cuda.is_available() else False

os.makedirs('acgan_images', exist_ok=True)

# Configure data loader
os.makedirs('../data/cifar10/', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/cifar10/', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ])),
    batch_size=batch_size, shuffle=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

        
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.nz = nz
        
        self.label_emb = nn.Embedding(10, 100)
        
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        x = gen_input.view(gen_input.size(0), -1, 1, 1)
        output = self.conv_block(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, ndf, 4, 1, 0, bias=False),
        )
        
        self.adv_layer = nn.Sequential( nn.Linear(ndf, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(ndf, num_classes)) ## 이 부분
        
    def forward(self, img):
        output = self.conv_block(img)
        flat6 = output.view(-1, 64)
        realfake = self.adv_layer(flat6)
        classes = self.aux_layer(flat6)
        return realfake, classes
      
generator = Generator(nz)
discriminator = Discriminator(num_classes)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        
        batch_size = imgs.shape[0]
        #print(batch_size)
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        real_ = Variable(torch.ones(labels.size()))
        
        #--------------------
        # Train Discriminator
        #--------------------
        
        discriminator.zero_grad()
        
        # Sample noise and labels as discriminator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))  # [64 x 100] 짜리 0 ~ 1 random noise
        gen_labels = Variable(LongTensor(np.random.randint(0, num_classes, batch_size))) # [64] 짜리 0 ~ 9 random int value
        fake_ = Variable(torch.zeros(gen_labels.size()))
        
        gen_imgs = generator(z, gen_labels)
        
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + (auxiliary_loss(real_aux, labels))) / 2
        
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + (auxiliary_loss(fake_aux, gen_labels))) / 2
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1)== gt)
        
        d_loss.backward()
        optimizer_D.step()
        
        #------------------
        # Train Generator
        #------------------
        
        generator.zero_grad()
        
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
        
        g_loss.backward()
        optimizer_G.step()
        
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, n_epochs, i, len(dataloader),
                                                            d_loss.item(), 100 * d_acc,
                                                            g_loss.item()))
