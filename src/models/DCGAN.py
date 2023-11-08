import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) implementation.

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

    def forward(self, batch_size):
        return self.Generator(batch_size)


class Generator(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) Generator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
        channels (`int`, *optional*, default to `3`): The channel of a input.
        device (`str`, *optional*, default to `cpu`): one of ['cpu', 'cuda', 'mps'].
    """

    def __init__(
        self,
        z_dim: int = 100,
        channels: int = 3,
        device: str = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.device = device

        conv_net = [
            nn.ConvTranspose2d(
                self.z_dim, 64 * 8, 4, 1, 0, bias=False
            ),  # (B, 64*8, 4, 4)
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),  # (B, 64*4, 8, 8)
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),  # (B, 64*2, 16, 16)
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),  # (B, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(
                64, self.channels, 4, 2, 1, bias=False
            ),  # (B, 3, 64, 64)
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

    def sample(self, batch_size):
        z = torch.randn(
            (batch_size, self.z_dim, 1, 1), device=self.device
        )  # (B, 100, 1, 1)
        return z

    def forward(self, batch_size):
        z = self.sample(batch_size)  # sample from noraml dist.
        output = self.conv_net(z)  # conv forward

        return output


class Discriminator(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) Discriminator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
        channels (`int`, *optional*, default to `3`): The channel of a input.
    """

    def __init__(
        self,
        channels: int = 3
    ):
        super().__init__()
        self.channels = channels

        conv_net = [
            nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False),  # (B, 64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),  # (B, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),  # (B, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),  # (B, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),  # (B, 1, 1, 1)
            nn.Sigmoid(),
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

    def forward(self, input):
        features = self.conv_net(input)  # conv forward
        output = features.flatten(start_dim=1) # reshape (B, 1, 1, 1) -> (B)

        return output
