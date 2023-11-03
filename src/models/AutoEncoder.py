import torch
import torch.nn as nn
import torch.nn.functional as F


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
        batch_size (`int`, *optional*, default to `64`): The size of mini-batch of input.
        channels (`int`, *optional*, default to `1`): Channels of input image.
    """

    def __init__(
        self,
        batch_size: int = 64,
        channels: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels

        conv_net = [
            nn.Conv2d(self.channels, 32, 3, 2, 1), # (B, 32, 14, 14)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.Conv2d(32, 64, 3, 2, 1), # (B, 64, 7, 7)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 4, 4)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.Conv2d(64, 64, 3, 2, 1), # (B, 64, 2, 2)
            nn.LeakyReLU(True),
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
        batch_size (`int`, *optional*, default to `64`): The size of mini-batch of input.
        channels (`int`, *optional*, default to `1`): Channels of input image.
    """

    def __init__(
        self, 
        batch_size: int = 64, 
        channels: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels

        conv_net = [
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # (B, 64, 4, 4)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64, 64, 3, 2, 1),  # (B, 64, 7, 7)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 32, 14, 14)
            nn.LeakyReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # (B, 1, 28, 28)
            nn.LeakyReLU(True),
            # -----------------------------
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
