import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ACGAN(nn.Module):
    r"""
    A modified ACGAN (Odena, A., Olah, C., & Shlens, J. (2017, July). Conditional image synthesis with auxiliary classifier gans. 
    In International conference on machine learning (pp. 2642-2651). PMLR.) implementation.

    Parameters:
        Generator (`nn.Module`): The Generator of GAN.
        Discriminator (`nn.Module`): The Discriminator of GAN.
    """

    def __init__(self, Generator: nn.Module, Discriminator: nn.Module):
        super().__init__()
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.device = self.Generator.device

    def get_generator(self):
        return self.Generator

    def get_discriminator(self):
        return self.Discriminator

    def forward(self, label: int):
        label = torch.tensor([label], device=self.device)
        return self.Generator(label)


class Generator(nn.Module):
    r"""
    A modified ACGAN (Odena, A., Olah, C., & Shlens, J. (2017, July). Conditional image synthesis with auxiliary classifier gans. 
    In International conference on machine learning (pp. 2642-2651). PMLR.) Generator implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of output image [C, H, W].
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
    """

    def __init__(
        self,
        output_shape: List = [1, 28, 28],
        z_dim: int = 100,
        n_class: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        self.out_channels, self.out_height, self.out_width = output_shape
        self.z_dim = z_dim
        self.n_class = n_class
        self.device = device

        self.fc = nn.Linear(self.z_dim + self.n_class, 384)

        conv_net = [
            nn.ConvTranspose2d(
                384, 192, 4, 1, 0, bias=False
            ), # (B, 48*4, 4, 4)
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(
                192, 96, 4, 2, 1, bias=False
            ), # (B, 48*2, 8, 8)
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),  # (B, 48, 16, 16)
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(48, self.out_channels, 4, 2, 1, bias=False),  # (B, 1, 32, 32)
            nn.Tanh(),  # last activation layer [-1,1]
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

    def sample(self, labels):
        one_hot_vectors = torch.nn.functional.one_hot(labels, self.n_class)
        z = torch.randn(
            (one_hot_vectors.shape[0], self.z_dim), device=self.device
        )  # (B, 100)
        z = torch.hstack([one_hot_vectors, z])
        return z

    def forward(self, labels):
        z = self.sample(labels)  # one-hot class information + sample from noraml dist.
        z = self.fc(z)
        z = z.view(-1, 384, 1, 1)
        output = self.conv_net(z)  # conv forward

        return output


class Discriminator(nn.Module):
    r"""
    A modified ACGAN (Odena, A., Olah, C., & Shlens, J. (2017, July). Conditional image synthesis with auxiliary classifier gans. 
    In International conference on machine learning (pp. 2642-2651). PMLR.) Discriminator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
    """

    def __init__(
        self,
        input_shape: List = [1, 28, 28],
        n_class: int = 10,
    ):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = input_shape
        self.n_class = n_class

        conv_net = [
            nn.Conv2d(self.in_channels, 16, 3, 2, 1, bias=False),  # (B, 16, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # -----------------------------
            nn.Conv2d(16, 16 * 2, 3, 1, 1, bias=False),  # (B, 32, 16, 16)
            nn.BatchNorm2d(16 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # -----------------------------
            nn.Conv2d(16 * 2, 16 * 4, 3, 2, 1, bias=False),  # (B, 64, 8, 8)
            nn.BatchNorm2d(16 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # -----------------------------
            nn.Conv2d(16 * 4, 16 * 8, 3, 1, 1, bias=False),  # (B, 128, 8, 8)
            nn.BatchNorm2d(16 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # -----------------------------
            nn.Conv2d(16 * 8, 16 * 16, 3, 2, 1, bias=False),  # (B, 256, 4, 4)
            nn.BatchNorm2d(16 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
            # -----------------------------
            nn.Conv2d(16 * 16, 16 * 32, 3, 1, 1, bias=False),  # (B, 512, 4, 4)
            nn.BatchNorm2d(16 * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        ]

        self.discriminator = nn.Linear(4*4*512, 1)
        self.auxiliary_classifier = nn.Linear(4*4*512, self.n_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

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

    def forward(self, input):
        features = self.conv_net(input)  # conv forward
        features = features.view(-1, 4*4*512)  # reshape (B, 512, 4, 4) -> (B, 4*4*512)
        disc = self.discriminator(features)
        aux = self.auxiliary_classifier(features)
        classes = self.log_softmax(aux)
        real_fake = torch.sigmoid(disc).view(-1, 1).squeeze(1)

        return real_fake, classes