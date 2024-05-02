import torch
import numpy as np
from utils import epsilon, generate_px_coordinates

import torch.nn as nn


class AvgSqrDiff_of_validPixels(nn.Module):
    """
    Calculates the average squared difference of valid pixels between two images.
    """

    def __init__(self):
        super(AvgSqrDiff_of_validPixels, self).__init__()

    def forward(self, image0: torch.Tensor, image1: torch.Tensor, isvalid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            image0 (torch.Tensor): The first image.
            image1 (torch.Tensor): The second image.
            isvalid (torch.Tensor): A mask indicating which pixels are valid.

        Returns:
            torch.Tensor: The average squared difference of valid pixels.
        """
        error = (image0 - image1)**2 * isvalid
        return torch.sum(error, dim=[1,2,3]) / (epsilon + torch.sum(isvalid, dim=[1,2,3]))


class Po_to_Img(nn.Module):
    """
    Converts 3D points to image coordinates.
    """

    def __init__(self):
        super(Po_to_Img, self).__init__()

    def forward(self, po: torch.Tensor, cam_K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            po (torch.Tensor): 3D points.
            cam_K (torch.Tensor): Camera intrinsic matrix.
            R (torch.Tensor): Rotation matrix.
            t (torch.Tensor): Translation vector.

        Returns:
            torch.Tensor: Image coordinates.
        """
        in_cam = torch.einsum('bij,byxj->byxi', R, po) + t[:, None, None]
        in_img = torch.einsum('bij,bxyj->bxyi', cam_K, in_cam)

        return in_img[...,:2] / (in_img[...,2:] + epsilon), in_cam


class UV_diff(nn.Module):
    """
    Calculates the difference between image coordinates and pixel coordinates.
    """

    def __init__(self, strides: int):
        super(UV_diff, self).__init__()
        self.strides = strides

    def forward(self, x: torch.Tensor, coord_K: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Image coordinates.
            coord_K (torch.Tensor): Pixel coordinates.

        Returns:
            torch.Tensor: Difference between image coordinates and pixel coordinates.
        """
        u, v = generate_px_coordinates(x.shape[1:3], coord_K, self.strides)
        return x - torch.stack([u, v], dim=-1)


class D_diff(nn.Module):
    """
    Calculates the difference between 3D points and depth values.
    """

    def __init__(self):
        super(D_diff, self).__init__()

    def forward(self, po: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            po (torch.Tensor): 3D points.
            depth (torch.Tensor): Depth values.

        Returns:
            torch.Tensor: Difference between 3D points and depth values.
        """
        return po[...,2:] - depth[...,None]


class ToOmega(nn.Module):
    """
    Converts weights to covariance matrices.
    """

    def __init__(self):
        super(ToOmega, self).__init__()

    def forward(self, w: torch.Tensor, isvalid: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            w (torch.Tensor): Weights.
            isvalid (torch.Tensor, optional): A mask indicating which weights are valid. Defaults to None.

        Returns:
            torch.Tensor: Covariance matrices.
        """
        assert(w.shape[-1] in [1,4])

        if isvalid is not None:
            w = w * isvalid

        if w.shape[-1] == 1:
            return w**2

        if w.shape[-1] == 4:
            A1 = w[...,0]
            A2 = w[...,1]
            A3 = w[...,2]
            A4 = w[...,3]
            result_shape = [w.shape[0], w.shape[1], w.shape[2], 2, 2]
            return torch.reshape(torch.stack([A1*A1+A3*A3, A2*A1+A4*A3, A1*A2+A3*A4, A2*A2+A4*A4], dim=-1), result_shape)


class Avg_nllh(nn.Module):
    """
    Calculates the average negative log-likelihood.
    """

    def __init__(self, pixel_cap: int = 100):
        super(Avg_nllh, self).__init__()
        self.pixel_cap = pixel_cap

    def forward(self, Omega: torch.Tensor, diff: torch.Tensor, isvalid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            Omega (torch.Tensor): Covariance matrices.
            diff (torch.Tensor): Difference between predicted values and ground truth values.
            isvalid (torch.Tensor): A mask indicating which pixels are valid.

        Returns:
            torch.Tensor: The average negative log-likelihood.
        """
        assert(Omega.shape[-1] == diff.shape[-1])
        assert(Omega.shape[-1] in [1,2])

        left_part = self.left_part(Omega)
        chi2_part = self.chi2_part(Omega, diff)

        error = torch.minimum(left_part + 0.5 * chi2_part, self.pixel_cap) * isvalid[...,0]
        chi2error = chi2_part * isvalid[...,0]

        avg_error = torch.sum(error, dim=[1,2])
        avg_chi2error = torch.sum(chi2error, dim=[1,2])
        divisor = torch.sum(isvalid[...,0], dim=[1,2]) + epsilon
        return avg_error / divisor, avg_chi2error / divisor

    def left_part(self, Omega: torch.Tensor) -> torch.Tensor:
        """
        Calculates the left part of the negative log-likelihood.

        Args:
            Omega (torch.Tensor): Covariance matrices.

        Returns:
            torch.Tensor: The left part of the negative log-likelihood.
        """
        const_part = torch.tensor(np.log((2*np.pi)**(Omega.shape[-1])), dtype=torch.float32)

        if Omega.shape[-1] == 1:
            var_part_pre_log = torch.squeeze(Omega**2, dim=-1)
        if Omega.shape[-1] == 2:
            Omega = torch.tensor(Omega, dtype=torch.float32)
            Omega += torch.eye(Omega.shape[-1]) * epsilon * (1. + torch.max(Omega,dim=[-2,-1],keepdim=True)[0])
            var_part_pre_log = Omega[...,0,0] * Omega[...,1,1] - Omega[...,1,0] * Omega[...,0,1]

        var_part = torch.log(var_part_pre_log + epsilon)

        return 0.5 * (const_part - var_part)

    def chi2_part(self, Omega: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
        """
        Calculates the chi-squared part of the negative log-likelihood.

        Args:
            Omega (torch.Tensor): Covariance matrices.
            diff (torch.Tensor): Difference between predicted values and ground truth values.

        Returns:
            torch.Tensor: The chi-squared part of the negative log-likelihood.
        """
        if Omega.shape[-1] == 1:
            return torch.squeeze(diff**2 * Omega**2, dim=-1)

        if Omega.shape[-1] == 2:
            b1 = diff[...,0]
            b2 = diff[...,1]
            A1 = Omega[...,0,0]
            A2 = Omega[...,0,1]
            A3 = Omega[...,1,0]
            A4 = Omega[...,1,1]
            return b1**2 * A1 + b1 * b2 * (A2 + A3) + b2**2 * A4


class Seg_Loss(nn.Module):
    """
    Calculates the segmentation loss.
    """

    def __init__(self):
        super(Seg_Loss, self).__init__()

    def forward(self, sigmoid: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            sigmoid (torch.Tensor): Sigmoid output of the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The segmentation loss, percent accuracy, and foreground percent accuracy.
        """
        labels = labels[..., None].float() / 255.
        return (
            labels * -torch.log(sigmoid + epsilon) + (1. - labels) * -torch.log(1 - sigmoid + epsilon),  # loss
            1. - torch.mean(torch.abs(labels - (sigmoid > 0.5).float())),  # percent
            1. - torch.sum(labels * torch.abs(labels - (sigmoid > 0.5).float())) / (epsilon + torch.sum(labels))  # fg_percent
        )
