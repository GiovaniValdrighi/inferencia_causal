import torch
import pyro
import numpy as np
from torch.nn import Module
from torch.nn.functional import one_hot
from networks import Decoder, Encoder
from torch import tensor
from pyro.distributions import OneHotCategorical, Normal, Bernoulli
import matplotlib.pyplot as plt

def ind_from_att(color, shape, scale, orientation, posX, posY):
    if type(color) != int:
        color = int(color)
    if type(shape) != int:
        shape = int(shape)
    if type(scale) != int:
        scale = int(scale)
    if type(orientation) != int:
        orientation = int(orientation)
    if type(posX) != int:
        posX = int(posX)
    if type(posY) != int:
        posY = int(posY)
    return (color)*3*6*40*32*32 + (shape )*6*40*32*32 + (scale)*40*32*32 + (orientation)*32*32 + (posX)*32 + posY 

def dummy_from_label(label):
    sizes = [1, 3, 6, 40, 32, 32]
    dummy = []
    for i, length in enumerate(sizes):
        dummy.append(one_hot(tensor(label[:, i], dtype = torch.int64), int(length)))
    return torch.cat(dummy, -1).to(torch.float32)

def label_from_dummy(dummy):
    label = []
    label.append(0)
    label.append(dummy[1:4].max(0)[1])
    label.append(dummy[4:10].max(0)[1])
    label.append(dummy[10:50].max(0)[1])
    label.append(dummy[50:82].max(0)[1])
    label.append(dummy[82:114].max(0)[1])
    return label

class VAE(Module):
    '''
    Class that define the posterior distribution q(z|x) as the model 
    with the decoder and the prior distribution q(x|z) as the guide 
    using the encoder.
    
    Inputs:  
    :pimg_dim: dimension of image vector
    :label_dim: dimension of label vector
    :latent_dim: dimension of Z space, output
    '''
    def __init__(self, latents_sizes, latents_names, img_dim = 4096, label_dim = 114, latent_dim = 200, use_CUDA = False):
        super(VAE, self).__init__()
        #creating networks
        self.encoder = Encoder(img_dim, label_dim, latent_dim)
        self.decoder = Decoder(img_dim, label_dim, latent_dim)
        self.img_dim = img_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.latents_sizes = latents_sizes
        self.latents_names = latents_names
        if use_CUDA:
            self.cuda()
        self.use_CUDA = use_CUDA
  
    def label_variable(self, label):
        new_label = []
        options = {'device': label.device, 'dtype': label.dtype}
        for i, length in enumerate(self.latents_sizes):
            prior = torch.ones(label.shape[0], length, **options) / (1.0 *length)
            new_label.append(pyro.sample("label_" + str(self.latents_names[i]), 
                             OneHotCategorical(prior), 
                             obs = one_hot(tensor(label[:, i], dtype = torch.int64), int(length))))
        new_label = torch.cat(new_label, -1)
        return new_label.to(torch.float32).to(label.device)

    def model(self, img, label):
        pyro.module("decoder", self.decoder)
        options = {'device': img.device, 'dtype': img.dtype}
        with pyro.plate("data", img.shape[0]):
            z_mean = torch.zeros(img.shape[0], self.latent_dim, **options)
            z_variance = torch.ones(img.shape[0], self.latent_dim, **options)
            z_sample = pyro.sample("latent", Normal(z_mean, z_variance).to_event(1))
            image = self.decoder.forward(z_sample, self.label_variable(label))
            pyro.sample("obs", Bernoulli(image).to_event(1), obs = img)


    def guide(self, img, label):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", img.shape[0]):
            z_mean, z_variance = self.encoder.forward(img, self.label_variable(label))
            pyro.sample("latent", Normal(z_mean, z_variance).to_event(1))
            
    def run_img(self, img, label, num = 1):
        label = label.reshape(1, -6)
        dummy_label = dummy_from_label(label)
        img = tensor(img.reshape(-1, 4096)).to(torch.float32)
        mean, var = self.encoder.forward(img, dummy_label)

        fig = plt.figure(figsize = (4, num*5))
        plots = []
        plots.append(plt.subplot(num+1, 1, 1))
        plots[0].set_title('Original image')
        plt.imshow(img.reshape(64, 64))
  
        for i in range(1, num):
            z_sample = Normal(mean, var).sample()
            vae_img = self.decoder.forward(z_sample, dummy_label)
            plots.append(plt.subplot(num+1, 1, i+1))
            plots[-1].set_title(str(i) +' - sample of latent space')
            plt.imshow(vae_img.detach().numpy().reshape(64, 64))
        plt.show()

    def change_attribute(self, img, label, attribute = 1):
        print('Attribute changed was ' + str(self.latents_names[attribute]))
        label = label.reshape(1, -6)
        new_label  = np.copy(label)
        while (new_label == label).all():
            val = np.random.choice(list(range(self.latents_sizes[attribute])))
            new_label[0, attribute] = val
        dummy_label = dummy_from_label(label)
        new_dummy = dummy_from_label(new_label)
        img = tensor(img.reshape(-1, 4096)).to(torch.float32)
        mean, var = self.encoder.forward(img, dummy_label)
  
        fig = plt.figure(figsize = (4, 15))
        plots = []
        
        plots.append(plt.subplot(3, 1, 1))
        plots[0].set_title('Original image')
        plt.imshow(img.reshape(64, 64))
        
        z_sample = Normal(mean, var).sample()
        vae_img = self.decoder.forward(z_sample, dummy_label)
        plots.append(plt.subplot(3, 1, 2))
        plots[1].set_title('Sample with original attribute')
        plt.imshow(vae_img.detach().numpy().reshape(64, 64))
        
        z_sample = Normal(mean, var).sample()
        vae_img = self.decoder.forward(z_sample, new_dummy)
        plots.append(plt.subplot(3, 1, 3))
        plots[2].set_title('Sample with changed attribute')
        plt.imshow(vae_img.detach().numpy().reshape(64, 64))
        plt.show()

