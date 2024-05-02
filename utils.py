import torch
import numpy as np
from typing import Tuple

epsilon = 0.00001

def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the norm of a tensor along the last axis.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Norm of the input tensor.
    """
    return torch.linalg.norm(x, axis=-1, keepdims=True)

def normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor along the last axis.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    return x / (norm(x) + epsilon)

def cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross product between two tensors.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Cross product of the input tensors.
    """
    return torch.linalg.cross(x, y)

def cross_n(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized cross product between two tensors.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Normalized cross product of the input tensors.
    """
    return normalize(cross(x, y))

def angle_between(x: torch.Tensor, y: torch.Tensor, dot_product: str = 'i, bxyi->bxy') -> torch.Tensor:
    """
    Compute the angle between two tensors.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.
        dot_product (str, optional): Dot product equation. Defaults to 'i, bxyi->bxy'.

    Returns:
        torch.Tensor: Angle between the input tensors.
    """
    numerator = torch.einsum(dot_product, normalize(x), normalize(y))
    return torch.math.acos(torch.minimum(torch.maximum(numerator, epsilon - 1.), 1. - epsilon))


def get_Angle_around_Axis(axis: torch.Tensor, v_from: torch.Tensor, v_to: torch.Tensor, dot_product: str = 'bxyi, bxyi->bxy') -> torch.Tensor:
    """
    Compute the angle around an axis between two vectors.

    Args:
        axis (torch.Tensor): Axis vector.
        v_from (torch.Tensor): Starting vector.
        v_to (torch.Tensor): Ending vector.
        dot_product (str, optional): Dot product equation. Defaults to 'bxyi, bxyi->bxy'.

    Returns:
        torch.Tensor: Angle around the axis between the vectors.
    """
    corrected_v_from = cross_n(cross(axis, v_from), axis)
    corrected_v_to = cross_n(cross(axis, v_to), axis)
    
    angle = angle_between(corrected_v_from, corrected_v_to, dot_product=dot_product)
    
    new_axis = cross_n(corrected_v_from, corrected_v_to)
    sign_correction_factor = torch.squeeze(torch.sign(torch.no_grad(norm(new_axis + axis) - 1.)), axis=-1)
    
    angle *= torch.minimum(sign_correction_factor * 2. + 1, 1)
    return angle


def change_Angle_around_Axis(axis: torch.Tensor, x: torch.Tensor, v_zero: torch.Tensor, factor: float, dot_product: str = 'bxyi, bxyi->bxy') -> torch.Tensor:
    """
    Change the angle around an axis between two vectors.

    Args:
        axis (torch.Tensor): Axis vector.
        x (torch.Tensor): Input vector.
        v_zero (torch.Tensor): Reference vector.
        factor (float): Scaling factor for the angle change.
        dot_product (str, optional): Dot product equation. Defaults to 'bxyi, bxyi->bxy'.

    Returns:
        torch.Tensor: Transformed vector.
    """
    factor = factor if not np.isinf(factor) else 0
    
    current_angle = get_Angle_around_Axis(axis, v_zero, x, dot_product=dot_product)
    angle_change = current_angle * (factor - 1) 
    R_to_make_newX_from_X = make_R_from_angle_axis(angle_change, axis)
    return torch.squeeze(torch.matmul(R_to_make_newX_from_X, x.unsqueeze(-1)), axis=-1)


def make_R_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix from an angle and an axis.

    Args:
        angle (torch.Tensor): Rotation angle.
        axis (torch.Tensor): Rotation axis.

    Returns:
        torch.Tensor: Rotation matrix.
    """
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    c = torch.math.cos(angle)
    s = torch.math.sin(angle)
    t = 1. - c
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]

    part_one = c[..., None, None] * torch.eye(3, dtype=c.dtype)

    part_two = t[..., None, None] * torch.stack([
        torch.stack([x * x, x * y, x * z], axis=-1),
        torch.stack([x * y, y * y, y * z], axis=-1),
        torch.stack([x * z, y * z, z * z], axis=-1)
    ], axis=-2)

    zero = torch.zeros_like(z)
    part_three = s[..., None, None] * torch.stack([
         torch.stack([zero, -z, y], axis=-1),
         torch.stack([z, zero, -x], axis=-1),
         torch.stack([-y, x, zero], axis=-1)
    ], axis=-2)

    return part_one + part_two + part_three   


def generate_px_coordinates(shape: Tuple[int, int], coord_K: torch.Tensor, strides: int = 1) -> torch.Tensor:
    """
    Generate pixel coordinates.

    Args:
        shape (tuple): Shape of the output coordinates.
        coord_K (torch.Tensor): Coordinate scaling factors.
        strides (int, optional): Stride value. Defaults to 1.

    Returns:
        torch.Tensor: Generated pixel coordinates.
    """
    u, v = torch.meshgrid(torch.arange(shape[1], dtype=torch.float32), torch.arange(shape[0], dtype=torch.float32))
    return u * coord_K[:, 0:1, 0:1] * strides + coord_K[:, 1:2, 0:1], v * coord_K[:, 0:1, 1:2] * strides + coord_K[:, 1:2, 1:2]
