import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3d -> GroupNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    """Downsampling block using Conv3d + GroupNorm + ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(out_channels, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_conv(x)


class SDVN(nn.Module):
    """Simple 3D UNet-like architecture"""

    def __init__(self):
        super().__init__()

        self.encoder_1 = nn.Sequential(
            DoubleConv(1, 16),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.down_conv1 = DownSample(16, 32)

        self.bridge = nn.Sequential(
            DoubleConv(32, 64),
            nn.GroupNorm(64, 64),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.upsample = nn.Sequential(
            DoubleConv(64, 32),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=1),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.decoder_1 = nn.Sequential(
            DoubleConv(16, 16),
            nn.GroupNorm(16, 16),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.last_layer = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def pad(self, image, template):
        """Pad `image` to match the spatial dimensions of `template`"""
        pad_x = abs(image.size(3) - template.size(3))
        pad_y = abs(image.size(2) - template.size(2))
        pad_z = abs(image.size(4) - template.size(4))

        b_x, e_x = math.floor(pad_x / 2), math.ceil(pad_x / 2)
        b_y, e_y = math.floor(pad_y / 2), math.ceil(pad_y / 2)
        b_z, e_z = math.floor(pad_z / 2), math.ceil(pad_z / 2)

        return F.pad(image, (b_z, e_z, b_x, e_x, b_y, e_y))

    def forward(self, x):
        # Encoder
        x1 = self.encoder_1(x)
        x1_cat = x + x1  # residual connection
        x1_down = self.down_conv1(x1_cat)

        # Bridge
        features = self.bridge(x1_down)

        # Decoder
        x2 = self.upsample(features)
        x2 = self.pad(x2, x1_cat)
        x2_cat = x2 + x1_cat
        x2_feat = self.decoder_1(x2_cat)
        x2_last = x2_cat + x2_feat

        return self.last_layer(x2_last)


def load_network(weights_path=None):
    """
    Utility function to load the network.
    """
    model = SDVN()
    
    if weights_path is not None:
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Optional: return checkpoint info if needed (epoch, optimizer, loss)

    return model
