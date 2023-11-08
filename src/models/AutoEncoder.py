import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AutoEncoder(nn.Module):
    r"""
    An AutoEncoder implementation.

    Parameters:
        Encoder (`nn.Module`): The Encoder of AutoEncoder.
        Decoder (`nn.Module`): The Decoder of AutoEncoder.
    """

    def __init__(self, Encoder: nn.Module, Decoder: nn.Module):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def get_encoder(self):
        return self.Encoder

    def get_decoder(self):
        return self.Decoder

    def forward(self, x):
        z = self.Encoder.forward(x)
        output = self.Decoder.forward(z)
        return output


class Encoder(nn.Module):
    r"""
    An Auto Encoder (Encoder) implementation.

    Parameters:
        input_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of input image [C, H, W].
    """

    def __init__(
        self,
        input_shape: List = [1, 28, 28]
    ):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = self.input_shape[0]

        conv_net = [
            nn.Conv2d(self.in_channels, 32, 3, 2, 1), # (B, 32, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(32, 64, 3, 2, 1), # (B, 64, 7, 7)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 2, 2)
            nn.LeakyReLU(0.2, inplace=True),
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
            if classname.find("Conv") != -1:  # Conv weight init
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:  # BatchNorm weight init
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        z = self.conv_net(x)  # conv forward

        return z


class Decoder(nn.Module):
    r"""
    An Auto Encoder (Decoder) implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of output image [C, H, W].
    """

    def __init__(
        self, 
        output_shape: List = [1, 28, 28]
    ):
        super().__init__()
        self.output_shape = output_shape
        self.out_channels = self.output_shape[0]

        conv_net = [
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.ConvTranspose2d(64, 64, 3, 2, 1),  # (B, 64, 7, 7)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 32, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
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
        reconstructed = self.conv_net(z)  # conv forward

        return reconstructed
