import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import List


class VariationalAutoEncoder(nn.Module):
    r"""
    An VariationalAutoEncoder (Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." 
    stat 1050 (2014): 1.) implementation.

    Parameters:
        Encoder (`nn.Module`): The Encoder of VariationalAutoEncoder.
        Decoder (`nn.Module`): The Decoder of VariationalAutoEncoder.
    """

    def __init__(self, Encoder: nn.Module, Decoder: nn.Module):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.z_dim = Encoder.z_dim

    def get_encoder(self):
        return self.Encoder

    def get_decoder(self):
        return self.Decoder
        
    def sample(self, mu, log_var):
        epsilon = torch.randn_like(mu, device=mu.device)
        z = mu + torch.exp(log_var/2)*epsilon
        return z

    def forward(self, x):
        mu, log_var = self.Encoder(x)
        z = self.sample(mu, log_var)
        output = self.Decoder(z)
        return output, mu, log_var
    
    def predict(self, device: str = 'cpu'):
        epsilon = torch.randn((1, self.z_dim, 1, 1), device=device)
        output = self.Decoder(epsilon)
        return output


class Encoder(nn.Module):
    r"""
    An VariationalAutoEncoder (Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." 
    stat 1050 (2014): 1.) Encoder implementation.

    Parameters:
        input_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of input image [C, H, W].
        z_dim (`int`, *optional*, default to `64`): The size of latents (mu, log_var, z).
    """

    def __init__(
        self,
        input_shape: List = [1, 28, 28],
        z_dim: int = 64
    ):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]
        self.z_dim = z_dim

        conv_net = [
            nn.Conv2d(self.in_channels, 32, 3, 2, 1), # (B, 32, 14, 14)
            nn.LeakyReLU(),
            # -----------------------------
            nn.Conv2d(32, 64, 3, 2, 1), # (B, 64, 7, 7)
            nn.LeakyReLU(),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 4, 4)
            nn.LeakyReLU(),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 2, 2)
            nn.LeakyReLU(),
        ]

        # Concat all layers
        self.conv_net = nn.Sequential()
        for i, layer in enumerate(conv_net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.conv_net.add_module(layer_name, layer)

        self.conv_out = nn.Conv2d(64, self.z_dim*2, 3, 2, 1)

        # initialize parameters
        self.init_params()

    def init_params(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # init Linear
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            if classname.find("Conv") != -1:  # Conv weight init
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:  # BatchNorm weight init
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        latent = self.conv_net(x)  # conv forward
        mu, log_var = torch.chunk(self.conv_out(latent), 2, dim=1)

        return mu, log_var


class Decoder(nn.Module):
    r"""
    An VariationalAutoEncoder (Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes." 
    stat 1050 (2014): 1.) Decoder implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of output image [C, H, W].
        z_dim (`int`, *optional*, default to `64`): The size of latents (mu, log_var, z).
    """

    def __init__(
        self, 
        output_shape: List = [1, 28, 28],
        z_dim: int = 64
    ):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = self.output_shape[0]
        self.z_dim = z_dim

        self.conv_in = nn.ConvTranspose2d(self.z_dim, 64, 4, 2, 1)  # (B, 64, 2, 2)

        conv_net = [
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4)
            nn.LeakyReLU(),
            # -----------------------------
            nn.ConvTranspose2d(64, 64, 3, 2, 1),  # (B, 64, 7, 7)
            nn.LeakyReLU(),
            # -----------------------------
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 32, 14, 14)
            nn.LeakyReLU(),
            # -----------------------------
            nn.ConvTranspose2d(32, self.out_channels, 4, 2, 1),  # (B, 1, 28, 28)
            nn.Tanh() # last activation layer [-1,1]
        ]

        # Concat all layers
        self.conv_net = nn.Sequential()
        for i, layer in enumerate(conv_net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.conv_net.add_module(layer_name, layer)

        # initialize parameters
        self.init_params()

    def init_params(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # init Linear
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif classname.find("Conv") != -1:  # init Conv
                nn.init.kaiming_normal_(m.weight)
            elif classname.find("BatchNorm") != -1:  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = self.conv_in(z)
        reconstructed = self.conv_net(z)  # conv forward

        return reconstructed

class VAE_KLLoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, mu: Tensor, log_var: Tensor) -> Tensor:
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1)
        return torch.mean(kl_loss) # reduction: mean
    
class VAELoss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.mse_loss = nn.MSELoss()
        self.vae_kl_loss = VAE_KLLoss()

    def forward(self, input: Tensor, target: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
        reconstruction_loss = self.mse_loss(input, target)
        vae_kl_loss = self.vae_kl_loss(mu, log_var)

        losses = {
            'MSELoss': reconstruction_loss,
            'KLLoss': vae_kl_loss,
            'TotalLoss': reconstruction_loss + vae_kl_loss * 0.001
        }

        return losses
        
