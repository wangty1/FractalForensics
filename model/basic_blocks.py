import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """ Convolutional block: Conv -> BN -> (Optional Activation). """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.01, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = torch.sigmoid(self.fc(self.avg_pool(x).view(b, c)))
        return x * attn.view(b, c, 1, 1)  # Scale original input


class SEResBlock(nn.Module):
    """SE-ResBlock: Residual Block with Squeeze-and-Excitation"""

    def __init__(self, channels):
        super(SEResBlock, self).__init__()
        self.conv = ConvBlock(channels, channels)
        self.se = SEBlock(channels)
        self.activation = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.se(out)
        return self.activation(out + residual)
