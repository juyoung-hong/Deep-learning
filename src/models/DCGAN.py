import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) implementation.

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
        return self.Generator.forward()


class Generator(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) Generator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
    """

    def __init__(
        self,
        z_dim: int = 100,
        batch_size: int = 64,
        channels: int = 3,
        height: int = 32,
        width: int = 32,
        device: str = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.device = device

        conv_net = [
            nn.ConvTranspose2d(
                self.z_dim, 64 * 8, 4, 1, 0, bias=False
            ),  # (B, 64*8, 2, 2)
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),  # (B, 64*4, 4, 4)
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),  # (B, 64*2, 8, 8)
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # -----------------------------
            nn.ConvTranspose2d(
                64, self.channels, 4, 2, 1, bias=False
            ),  # (B, 3, 32, 32)
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

    def sample(self):
        batch_size = self.batch_size if self.training else 1
        z = torch.randn(
            (batch_size, self.z_dim, 1, 1), device=self.device
        )  # (B, 100, 1, 1)
        return z

    def forward(self):
        z = self.sample()  # sample from noraml dist.
        output = self.conv_net(z)  # conv forward

        return output


class Discriminator(nn.Module):
    r"""
    A modified DCGAN (Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with
    deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.) Discriminator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
    """

    def __init__(
        self, batch_size: int = 64, channels: int = 3, height: int = 32, width: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        conv_net = [
            nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False),  # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),  # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),  # (B, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),  # (B, 512, 2, 2)
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
        output = features.view(-1)  # reshape (B, 1, 1, 1) -> (B)

        return output
