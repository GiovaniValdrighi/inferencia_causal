import torch
import pyro
import numpy as np
from torch.nn import Module
from networks import Decoder, Encoder
from torch import tensor
from pyro.distributions import OneHotCategorical, Normal, Bernoulli

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
                             obs = torch.nn.functional.one_hot(tensor(label[:, i], dtype = torch.int64), int(length))))
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
