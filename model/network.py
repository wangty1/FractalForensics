import torch
import torch.nn as nn

from .basic_blocks import ConvBlock, SEResBlock


class ImageFeatureExtractor(nn.Module):
    """ Extracts features from the input image. """

    def __init__(self, out_channels=64, num_blocks=3):
        super(ImageFeatureExtractor, self).__init__()
        self.conv_head = ConvBlock(3, out_channels, kernel_size=7, stride=1, padding=3)
        self.resblocks = nn.Sequential(*[
            SEResBlock(out_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.conv_head(x)
        x = self.resblocks(x)
        return x


class WatermarkDiffusion(nn.Module):
    """ Handles the entire watermark processing pipeline. """

    def __init__(self, img_size, wtm_size, out_channels=64, num_blocks=3):
        super().__init__()
        self.img_size = img_size
        self.wtm_size = wtm_size
        self.patch_size = img_size // wtm_size

        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='nearest')

        self.watermark_feature_extractor = nn.Sequential(
            ConvBlock(4, 32),
            *[SEResBlock(32) for _ in range(num_blocks)],
            ConvBlock(32, out_channels)
        )

    def forward(self, watermark):
        """Processes the watermark and applies diffusion."""
        expanded = self.upsample(watermark)
        wtm_feature = self.watermark_feature_extractor(expanded)
        return wtm_feature


class FractalForensics(nn.Module):
    """ Full Watermark Embedding & Image Reconstruction Pipeline. """

    def __init__(self, img_size, wtm_size, latent_channels=64, img_blocks=3, wtm_blocks=3, rec_blocks=4):
        super(FractalForensics, self).__init__()
        self.img_size = img_size
        self.wtm_size = wtm_size
        self.latent_channels = latent_channels

        self.image_extractor = ImageFeatureExtractor(
            out_channels=latent_channels,
            num_blocks=img_blocks
        )
        self.watermark_diffuser = WatermarkDiffusion(
            img_size=img_size,
            wtm_size=wtm_size,
            out_channels=latent_channels,
            num_blocks=wtm_blocks
        )

        intermediate_channels = latent_channels * 3 // 2
        self.fusion = nn.Sequential(
            ConvBlock(latent_channels * 2, intermediate_channels),
            ConvBlock(intermediate_channels, latent_channels),
        )
        self.resblocks = nn.Sequential(*[
            SEResBlock(latent_channels) for _ in range(rec_blocks)
        ])

        self.recon = nn.Sequential(
            ConvBlock(latent_channels + 3, 32),
            ConvBlock(32, 3, activation=False)
        )

    def forward(self, img, wtm):
        img_features = self.image_extractor(img)
        wtm_features = self.watermark_diffuser(wtm)

        fused = torch.cat([img_features, wtm_features], dim=1)
        fused = self.fusion(fused)
        fused = self.resblocks(fused)

        fused_cat = torch.cat([fused, img], dim=1)
        rec_img = self.recon(fused_cat)
        return torch.clamp(rec_img, min=-1, max=1)


class WatermarkDecoder(nn.Module):
    """ Decodes the embedded fractal watermark from the watermarked image. """

    def __init__(self, latent_channels=64, num_blocks=3):
        super(WatermarkDecoder, self).__init__()
        self.conv_head = nn.Sequential(
            ConvBlock(3, latent_channels // 2, kernel_size=5, padding=2),
            nn.Dropout2d(p=0.2),
            ConvBlock(latent_channels // 2, latent_channels, stride=2),
            nn.Dropout2d(p=0.2),
            ConvBlock(latent_channels, latent_channels, stride=2)
        )

        self.resblocks = nn.Sequential(*[
            SEResBlock(latent_channels) for _ in range(num_blocks)
        ])

        self.conv_tail = nn.Sequential(
            ConvBlock(latent_channels, latent_channels, stride=2),
            nn.Dropout2d(p=0.1),
            ConvBlock(latent_channels, latent_channels // 2, stride=2),
            nn.Dropout2d(p=0.05),
            ConvBlock(latent_channels // 2, latent_channels // 2, stride=2),
            nn.Conv2d(latent_channels // 2, 4, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_head(x)
        x = self.resblocks(x)
        x = self.conv_tail(x)
        return x
