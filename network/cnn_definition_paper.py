import torch
from network.augmentations import *

import torch.nn as nn
import torch.nn.functional as F



def dense_block(x, channels, iterations):
    """
    A helper function that implements a dense block.

    Args:
        x (torch.Tensor): Input tensor.
        channels (int): Number of input and output channels.
        iterations (int): Number of iterations for the dense block.

    Returns:
        torch.Tensor: Output tensor after passing through the dense block.
    """
    for i in range(iterations):
        x1 = x
        x1 = nn.Conv2d(channels, channels, kernel_size=1, padding='same')(x1)
        x1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')(x1)
        x = torch.cat([x, x1], dim=1)
    return nn.Conv2d(channels * 2, channels * 2, kernel_size=1, padding='same')(x)


def rgb255_to_obj_net(rgb):
    """
    Converts an RGB image to object network outputs.

    Args:
        rgb (numpy.ndarray): Input RGB image.

    Returns:
        tuple: A tuple containing the outputs of the object network:
            - star (torch.Tensor): Output tensor for star detection.
            - dash (torch.Tensor): Output tensor for dash detection.
            - w_px (torch.Tensor): Output tensor for width prediction.
            - w_d (torch.Tensor): Output tensor for width detection.
            - seg (torch.Tensor): Output tensor for segmentation.
    """
    input_x_ = ContrastNoise(0.25)(rgb)
    input_x_ = ContrastNoiseSingle(0.25)(input_x_)
    input_x_ = GaussianNoise(.08 * 128)(input_x_)
    input_x_ = BrightnessNoise(0.2 * 128)(input_x_)

    rgb = torch.tensor(rgb).permute(0, 3, 1, 2)  # Assuming rgb is a numpy array

    x = nn.Conv2d(1, 8, kernel_size=5, padding='same')(rgb)
    tier0 = x

    x = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding='same')(x)
    x = dense_block(x, 32, 3)
    tier1 = x

    x = nn.Conv2d(16, 64, kernel_size=5, stride=2, padding='same')(x)
    x = dense_block(x, 64, 6)
    x = dense_block(x, 64, 6)
    tier2 = x

    x = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding='same')(x)
    x = dense_block(x, 64, 12)
    x = dense_block(x, 64, 12)
    tier3 = x

    x = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding='same')(x)
    x = dense_block(x, 128, 12)

    def up_path(x):
        """
        Implements the up path of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the up path.
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, tier3], dim=1)
        x = nn.Conv2d(192, 64, kernel_size=3, padding='same')(x)
        x = dense_block(x, 32, 12)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, tier2], dim=1)
        x = nn.Conv2d(96, 32, kernel_size=3, padding='same')(x)
        x = dense_block(x, 16, 6)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, tier1], dim=1)
        x = nn.Conv2d(48, 24, kernel_size=3, padding='same')(x)
        x = dense_block(x, 12, 4)

        return x

    star_x = up_path(x)
    star = nn.Conv2d(48, 3, kernel_size=1, padding='same')(star_x)

    dash_x = up_path(x)
    dash = nn.Conv2d(48, 3, kernel_size=1, padding='same')(dash_x)

    w_px_x = up_path(x)
    w_px = nn.Conv2d(48, 4, kernel_size=1, padding='same')(w_px_x)

    w_d_x = up_path(x)
    w_d = nn.Conv2d(48, 1, kernel_size=1, padding='same')(w_d_x)

    seg_x = up_path(x)
    seg = nn.Conv2d(48, 1, kernel_size=1, padding='same')(seg_x)
    seg = torch.sigmoid(seg)

    return star, dash, w_px, w_d, seg
