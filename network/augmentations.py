import torch

import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise to the input tensor during training.

    Args:
        stddev (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Tensor with added Gaussian noise.
    """
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs):
        """
        Forward pass of the GaussianNoise module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        """
        if self.training:
            noise = torch.randn_like(inputs) * self.stddev
            return torch.clamp(inputs + noise, 0., 255.)
        else:
            return inputs
    
class ContrastNoiseSingle(nn.Module):
    """
    Adds contrast noise to each channel of the input tensor during training.

    Args:
        stddev (float): Standard deviation of the contrast noise.

    Returns:
        torch.Tensor: Tensor with added contrast noise.
    """
    def __init__(self, stddev):
        super(ContrastNoiseSingle, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs):
        """
        Forward pass of the ContrastNoiseSingle module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added contrast noise.
        """
        if self.training:
            noise = torch.empty_like(inputs)
            for i in range(inputs.size(0)):
                noise[i, :, :, 0] = torch.clamp(torch.randn_like(inputs[i, :, :, 0]) * self.stddev, -self.stddev, self.stddev)
                noise[i, :, :, 1] = torch.clamp(torch.randn_like(inputs[i, :, :, 1]) * self.stddev, -self.stddev, self.stddev)
                noise[i, :, :, 2] = torch.clamp(torch.randn_like(inputs[i, :, :, 2]) * self.stddev, -self.stddev, self.stddev)
            return torch.cat([inputs[:, :, :, 0:1] + noise[:, :, :, 0:1],
                              inputs[:, :, :, 1:2] + noise[:, :, :, 1:2],
                              inputs[:, :, :, 2:3] + noise[:, :, :, 2:3]], dim=-1)
        else:
            return inputs
    
class ContrastNoise(nn.Module):
    """
    Adds contrast noise to the input tensor during training.

    Args:
        stddev (float): Standard deviation of the contrast noise.

    Returns:
        torch.Tensor: Tensor with added contrast noise.
    """
    def __init__(self, stddev):
        super(ContrastNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs):
        """
        Forward pass of the ContrastNoise module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added contrast noise.
        """
        if self.training:
            return torch.clamp(inputs + torch.randn_like(inputs) * self.stddev, 0., 255.)
        else:
            return inputs
    
class BrightnessNoise(nn.Module):
    """
    Adds brightness noise to the input tensor during training.

    Args:
        stddev (float): Standard deviation of the brightness noise.

    Returns:
        torch.Tensor: Tensor with added brightness noise.
    """
    def __init__(self, stddev):
        super(BrightnessNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs):
        """
        Forward pass of the BrightnessNoise module.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added brightness noise.
        """
        if self.training:
            return torch.clamp(inputs + torch.randn_like(inputs) * self.stddev, 0., 255.)
        else:
            return inputs
