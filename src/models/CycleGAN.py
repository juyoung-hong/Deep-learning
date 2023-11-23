import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class CycleGAN(nn.Module):
    r"""
    A modified CycleGAN (Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." 
    Proceedings of the IEEE international conference on computer vision. 2017.) implementation.

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

class DownSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),  
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.module(x)
    
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),  
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),  
            nn.InstanceNorm2d(out_channels),
        )
    
    def forward(self, x):
        output = self.module(x) + x
        return output

class UpSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.module(x)


class Generator(nn.Module):
    r"""
    A modified CycleGAN (Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." 
    Proceedings of the IEEE international conference on computer vision. 2017.) Generator implementation.

    Parameters:
        output_shape (`List`, *optional*, default to `[3, 128, 128]`): Shape of output image [C, H, W].
    """

    def __init__(
        self,
        output_shape: List = [3, 128, 128],
    ):
        super().__init__()
        self.out_channels, self.out_height, self.out_width = output_shape

        # downsample blocks
        self.d1 = DownSample2D(self.out_channels, 64)
        self.d2 = DownSample2D(64, 128)
        self.d3 = DownSample2D(128, 256)

        # mid blocks
        self.r1 = ResidualBlock2D(256, 256)
        self.r2 = ResidualBlock2D(256, 256)
        self.r3 = ResidualBlock2D(256, 256)
        self.r4 = ResidualBlock2D(256, 256)
        self.r5 = ResidualBlock2D(256, 256)
        self.r6 = ResidualBlock2D(256, 256)
        self.r7 = ResidualBlock2D(256, 256)
        self.r8 = ResidualBlock2D(256, 256)
        self.r9 = ResidualBlock2D(256, 256)

        # upsample blocks
        self.u1 = UpSample2D(256, 128)
        self.u2 = UpSample2D(128, 64)
        self.u3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, self.out_channels, 3, 1, 1),
            nn.Tanh()
        )

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
        # down sample
        z1 = self.d1(x)
        z2 = self.d2(z1)
        z3 = self.d3(z2)

        # skip-connection
        z3 = self.r1(z3)
        z3 = self.r2(z3)
        z3 = self.r3(z3)
        z3 = self.r4(z3)
        z3 = self.r5(z3)
        z3 = self.r6(z3)
        z3 = self.r7(z3)
        z3 = self.r8(z3)
        z3 = self.r9(z3)
        
        # up sample
        z4 = self.u1(z3)
        z5 = self.u2(z4)
        output = self.u3(z5)

        return output

class DiscDownSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, strides=2, padding=1, norm=True):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),  
            (nn.InstanceNorm2d(out_channels) if norm else nn.Identity()),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        return self.module(x)

class Discriminator(nn.Module):
    r"""
    A modified CycleGAN (Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." 
    Proceedings of the IEEE international conference on computer vision. 2017.) Discriminator implementation.

    Parameters:
        input_shape (`List`, *optional*, default to `[3, 128, 128]`): Shape of input image [C, H, W].
    """

    def __init__(
        self,
        input_shape: List = [3, 128, 128],
    ):
        super().__init__()
        self.in_channels, self.in_height, self.in_width = input_shape

        conv_net = [
            DiscDownSample2D(self.in_channels, 64, norm=False),
            DiscDownSample2D(64, 128),
            DiscDownSample2D(128, 256),
            DiscDownSample2D(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, 1, 1)
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
        output = self.conv_net(input)  # conv forward

        return output
        