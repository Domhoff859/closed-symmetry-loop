import torch

import torch.nn as nn
import torch.nn.functional as F

class GaussianNoise(nn.Module):
    def __init__(self, stddev: float) -> None:
        super(GaussianNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(inputs) * self.stddev
            return torch.clamp(inputs + noise, 0., 255.)
        else:
            return inputs
    
class ContrastNoiseSingle(nn.Module):
    def __init__(self, stddev: float) -> None:
        super(ContrastNoiseSingle, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, stddev: float) -> None:
        super(ContrastNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.clamp(inputs + torch.randn_like(inputs) * self.stddev, 0., 255.)
        else:
            return inputs
    
class BrightnessNoise(nn.Module):
    def __init__(self, stddev: float) -> None:
        super(BrightnessNoise, self).__init__()
        self.stddev = stddev
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            return torch.clamp(inputs + torch.randn_like(inputs) * self.stddev, 0., 255.)
        else:
            return inputs
