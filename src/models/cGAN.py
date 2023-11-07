import torch
import torch.nn as nn
import torch.nn.functional as F


class cGAN(nn.Module):
    r"""
    A modified cGAN(Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." 
    arXiv preprint arXiv:1411.1784 (2014).) implementation.

    Parameters:
        Generator (`nn.Module`): The Generator of cGAN.
        Discriminator (`nn.Module`): The Discriminator of cGAN.
    """

    def __init__(self, Generator: nn.Module, Discriminator: nn.Module):
        super().__init__()
        self.Generator = Generator
        self.Discriminator = Discriminator

    def get_generator(self):
        return self.Generator

    def get_discriminator(self):
        return self.Discriminator

    def forward(self, batch_size, labels):
        return self.Generator(batch_size, labels)


class Generator(nn.Module):
    r"""
    A modified cGAN(Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." 
    arXiv preprint arXiv:1411.1784 (2014).) Generator implementation.

    Parameters:
        z_dim (`int`, *optional*, default to `100`): The size of noise as a input.
        channels (`int`, *optional*, default to `1`): The channel of a input.
        height (`int`, *optional*, default to `28`): The height of a input.
        width (`int`, *optional*, default to `28`): The width of a input.
        n_class (`int`, *optional*, default to `10`): The number of class label of cGAN.
        device (`str`, *optional*, default to `cpu`): one of ['cpu', 'cuda', 'mps'].
    """

    def __init__(
        self,
        z_dim: int = 100,
        channels: int = 1,
        height: int = 28,
        width: int = 28,
        n_class: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        self.z_dim = z_dim
        self.channels = channels
        self.height = height
        self.width = width
        self.device = device
        self.n_class = n_class

        self.embed = nn.Embedding(self.n_class, self.n_class)

        net = [
            nn.Linear(self.z_dim + self.n_class, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(1024, self.channels * self.width * self.height),
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

    def sample(self, batch_size):
        z = torch.randn(size=(batch_size, self.z_dim), device=self.device)
        return z

    def forward(self, batch_size, labels):
        z = self.sample(batch_size)  # sample from noraml dist.
        c = self.embed(labels) # embedding labels.
        x = torch.cat([z, c], axis=1)
        output = self.net(x)  # mlp forward
        output = output.view(
            -1, self.channels, self.height, self.width
        )  # reshape (B, 784) -> (B, C, H, W)

        return output


class Discriminator(nn.Module):
    r"""
    A modified cGAN(Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." 
    arXiv preprint arXiv:1411.1784 (2014).) Discriminator implementation.

    Parameters:
        channels (`int`, *optional*, default to `1`): The channel of a input.
        height (`int`, *optional*, default to `28`): The height of a input.
        width (`int`, *optional*, default to `28`): The width of a input.
        n_class (`int`, *optional*, default to `10`): The number of class label of cGAN.
    """

    def __init__(
        self,
        channels: int = 1, 
        height: int = 28, 
        width: int = 28, 
        n_class: int = 10
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.n_class =n_class

        self.embed = nn.Embedding(self.n_class, self.n_class)

        net = [
            nn.Linear(self.channels * self.width * self.height + self.n_class, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            # -----------------------------
            nn.Linear(512, 1),
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

    def forward(self, input, labels):
        x = input.flatten(start_dim=1) # reshape (B, 1, 28, 28) -> (B, 784)
        c = self.embed(labels)
        x = torch.cat([x, c], axis=1)
        output = self.net(x)  # mlp forward

        return output
