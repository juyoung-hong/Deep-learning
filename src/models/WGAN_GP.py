import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import List


class WGAN_GP(nn.Module):
    r"""
    A modified  WGAN-GP(Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." 
    Advances in neural information processing systems 30 (2017)) implementation.

    Parameters:
        Generator (`nn.Module`): The Generator of DCGAN.
        Discriminator (`nn.Module`): The Discriminator of DCGAN.
    """

    def __init__(self, Generator: nn.Module, Discriminator: nn.Module):
        super().__init__()
        self.Generator = Generator
        self.Discriminator = Discriminator

    def get_generator(self):
        return self.Generator

    def get_discriminator(self):
        return self.Discriminator

    def forward(self):
        return self.Generator(1)


class Generator(nn.Module):
    r"""
    A modified  WGAN-GP(Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." 
    Advances in neural information processing systems 30 (2017)) Generator implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of output image [C, H, W].
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
        device (`str`, *optional*, default to `cpu`): one of ['cpu', 'cuda', 'mps'].
    """

    def __init__(
        self,
        output_shape: List = [1, 28, 28],
        z_dim: int = 100,
        device: str = None,
    ):
        super().__init__()
        self.out_channels, self.out_height, self.out_width = output_shape
        self.z_dim = z_dim
        self.device = device

        mlp_net = [
            nn.Linear(self.z_dim, 8*16*2*2),
            nn.ReLU()
        ]

        conv_net = [
            nn.ConvTranspose2d(8 * 16, 4 * 16, 4, 2, 1),  # (B, 64*8, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(4 * 16),
            # -----------------------------
            nn.ConvTranspose2d(4 * 16, 2 * 16, 4, 2, 1),  # (B, 64*8, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(2 * 16),
            # -----------------------------
            nn.ConvTranspose2d(2 * 16, 16, 4, 2, 1),  # (B, 64*8, 4, 4)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # -----------------------------
            nn.ConvTranspose2d(16, self.out_channels, 4, 2, 1),  # (B, 64*8, 4, 4)
            nn.Tanh() # last activation layer [-1,1] 
        ]

        # Concat all layers
        self.mlp_net = nn.Sequential()
        for i, layer in enumerate(mlp_net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.mlp_net.add_module(layer_name, layer)

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
            if classname.find("Conv") != -1:  # Conv weight init
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:  # BatchNorm weight init
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def sample(self, batch_size: int):
        z = torch.randn(
            (batch_size, self.z_dim), device=self.device
        )  # (B, 100)
        return z

    def forward(self, batch_size: int):
        z = self.sample(batch_size)  # sample from noraml dist.
        x = self.mlp_net(z)
        x = x.view(-1, 8*16, 2, 2)
        output = self.conv_net(x)  # conv forward

        return output


class Critic(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) Discriminator implementation.

    Parameters:
        input_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of input image [C, H, W].
    """

    def __init__(
        self,
        input_shape: List = [1, 28, 28],
    ):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = input_shape

        conv_net = [
            nn.Conv2d(self.in_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(16, 16 * 2, 4, 2, 1),  # (B, 128, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(16 * 2, 16 * 4, 4, 2, 1),  # (B, 256, 8, 8)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(16 * 4, 16 * 8, 4, 2, 1),  # (B, 512, 4, 4)
            nn.Sigmoid(),
        ]

        mlp_net = [
            nn.Linear(8*16*2*2, 1),
            nn.Sigmoid()
        ]

        # Concat all layers
        self.conv_net = nn.Sequential()
        for i, layer in enumerate(conv_net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.conv_net.add_module(layer_name, layer)

        self.mlp_net = nn.Sequential()
        for i, layer in enumerate(mlp_net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.mlp_net.add_module(layer_name, layer)

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

    def forward(self, input):
        features = self.conv_net(input)  # conv forward
        features = features.flatten(start_dim=1) # reshape (B, 1, 1, 1) -> (B)
        output = self.mlp_net(features)

        return output