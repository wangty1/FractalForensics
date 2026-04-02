import torch.nn as nn

from .basic_blocks import ConvBlock, SEResBlock


class Discriminator(nn.Module):
    """ Discriminator for adversarial training to improve visual quality. """

    def __init__(self, latent_channels=64, num_blocks=3):
        super(Discriminator, self).__init__()
        self.conv_head = ConvBlock(3, latent_channels, stride=2)

        self.resblocks = nn.Sequential(*[
            SEResBlock(latent_channels) for _ in range(num_blocks)
        ])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(latent_channels, 1)
        )

    def forward(self, x):
        x = self.conv_head(x)
        x = self.resblocks(x)
        x = self.fc(x)
        return x
