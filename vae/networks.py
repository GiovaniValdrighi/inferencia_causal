import numpy as np
import torch
from torch.nn import Module, Linear, Softplus, Sigmoid

#torch.cat, torch.exp 
#np.random.permutation

def setup_data_loader(images, classes, batch_size = 128, use_CUDA = False):
  '''
    Function that receives two arrays, an array of the data images and an array of it latents values
    and generate a DataLoader for train and test data.
    
    Input:
    :images: array of size [*, 64, 64] with images data
    :classes: array of size [*, 6] with latent variables of the images
    
    Output:
    :train_loader: torch data loader with train data (images and latents)
    :test_loader: torch data loader with test data (images and latents)
  '''
  index = np.random.permutation(imgs.shape[0])
  images = images[index].astype(np.float32)
  classes = classes[index].astype(np.float32)
  train_df = torch.utils.data.TensorDataset(torch.from_numpy(images[100000:].reshape(-1, 4096)), torch.from_numpy(classes[100000:]))
  test_df = torch.utils.data.TensorDataset(torch.from_numpy(images[:100000].reshape(-1, 4096)), torch.from_numpy(classes[:100000]))
  kwargs = {'num_workers': 1, 'pin_memory': use_CUDA}
  train_loader = torch.utils.data.DataLoader(train_df, batch_size, shuffle = False, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_df, batch_size, shuffle = False, **kwargs)
  return train_loader, test_loader


class Encoder(Module):
  '''
    Class that receive image and its latent value and pass through the encoder neural network
    with 2 layers. One layer generates the hidden variable and the second layer generate
    the loc and scale of the z (latent space) gaussian variable.

    Input:
    :img_dim: dimension of image vector
    :label_dim: dimension of label vector
    :latent_dim: dimension of latent space, output

    Output:
    :loc_z: tensor of shape [200] with the loc of the gaussian distribution of Z variable
    :scale_z: tensor of shape [200] with the scale of the gaussian distribution of Z variable
  '''
  def __init__(self, img_dim = 4096, label_dim = 114, latent_dim = 200):
    super(Encoder, self).__init__()
    self.img_dim = img_dim
    self.label_dim = label_dim
    self.latent_dim = latent_dim 
    self.fc1 = Linear(img_dim + label_dim, 1000)
    self.fc21 = Linear(1000, latent_dim)
    self.fc22 = Linear(1000, latent_dim)
    self.softplus = Softplus()

  def forward(self, img, label):
    data = torch.cat((img, label), -1)
    hidden = self.softplus(self.fc1(data))
    loc_z = self.fc21(hidden)
    scale_z = torch.exp(self.fc22(hidden))
    return loc_z, scale_z
 

  class Decoder(Module):
  '''
    Class that receive the Z variable and its latent values and pass through the decode neural network
    with 2 layers. One layer generates the hidden variable and the second layer generate the image.

    Input:
    :img_dim: dimension of image vector
    :label_dim: dimension of label vector
    :latent_dim: dimension of latent space, output

    Output:
    :image: tensor of shape [4096] that is the reconstruction of the image data
    
  '''
  def __init__(self, img_dim = 4096, label_dim = 114, latent_dim = 200):
    super(Decoder, self).__init__()
    self.img_dim = img_dim
    self.label_dim = label_dim
    self.latent_dim = latent_dim
    self.fc1 = Linear(latent_dim+label_dim, 1000)
    self.fc2 = Linear(1000, img_dim)
    self.softplus = Softplus()
    self.sigmoid = Sigmoid()

  def forward(self, latent, label):
    data = torch.cat((latent, label), -1)
    hidden = self.softplus(self.fc1(data))
    image = self.sigmoid(self.fc2(hidden))
    return image
