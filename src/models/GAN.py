import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class GAN(nn.Module):
    r"""
    A modified GAN(Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.) implementation.

    Parameters:
        Generator (`nn.Module`): The Generator of GAN.
        Discriminator (`nn.Module`): The Discriminator of GAN.
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
    A modified GAN(Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.) Generator implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of output image [C, H, W].
        z_dim (`int`, *optional*, default to `64`): The size of noise as a input.
        device (`str`, *optional*, default to `cpu`): one of ['cpu', 'cuda', 'mps'].
    """

    def __init__(
        self,
        output_shape: List = [1, 28, 28],
        z_dim: int = 64,
        device: str = 'cpu'
    ):
        super().__init__()
        self.out_channels, self.out_height, self.out_width = output_shape
        self.z_dim = z_dim
        self.device = device

        net = [
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(256, self.out_channels * self.out_width * self.out_height),
            nn.Tanh(),  # last activation layer [-1,1]
        ]

        # Concat all layers
        self.net = nn.Sequential()
        for i, layer in enumerate(net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.net.add_module(layer_name, layer)

        # initialize parameters
        self.init_params()

    def init_params(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # init Linear
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif classname.find("BatchNorm") != -1:  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def sample(self, batch_size: int):
        z = torch.randn(size=(batch_size, self.z_dim), device=self.device)
        return z

    def forward(self, batch_size: int):
        z = self.sample(batch_size)  # sample from noraml dist.
        output = self.net(z)  # mlp forward
        output = output.view(
            -1, self.out_channels, self.out_height, self.out_width
        )  # reshape (B, 784) -> (B, C, H, W)

        return output


class Discriminator(nn.Module):
    r"""
    A modified GAN(Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020).
    Generative adversarial networks. Communications of the ACM, 63(11), 139-144.) Discriminator implementation.

    Parameters:
        input_shape (`List`, *optional*, default to `[1, 28, 28]`): Shape of input image [C, H, W].
    """

    def __init__(
        self, input_shape: List = [1, 28, 28],
    ):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = input_shape

        net = [
            nn.Linear(self.in_channels * self.in_width * self.in_height, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ]

        # Concat all layers
        self.net = nn.Sequential()
        for i, layer in enumerate(net):
            layer_name = f"{type(layer).__name__.lower()}_{i}"
            self.net.add_module(layer_name, layer)

        # initialize parameters
        self.init_params()

    def init_params(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:  # init Linear
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif classname.find("BatchNorm") != -1:  # init BN
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = input.flatten(start_dim=1) # reshape (B, 1, 28, 28) -> (B, 784)
        output = self.net(x)  # mlp forward

        return output
