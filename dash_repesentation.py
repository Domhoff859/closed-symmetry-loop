import torch
from utils import *

class DashRepresentation(torch.nn.Module):
    def __init__(self, offset: torch.Tensor):
        """
        Initializes the DashRepresentation module.

        Args:
            offset (torch.Tensor): The offset tensor.
        """
        super(DashRepresentation, self).__init__()
        self.offset = offset
        
    def forward(self, R: torch.Tensor, po: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the DashRepresentation module.

        Args:
            R (torch.Tensor): The R tensor.
            po (torch.Tensor): The po tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.einsum('bij,byxj->byxi', R, po) + self.offset
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the DashRepresentation module.

        Returns:
            dict: The configuration dictionary.
        """
        return {'offset': self.offset}


class RemoveCameraEffect(torch.nn.Module):
    def __init__(self, strides: int = 1):
        """
        Initializes the RemoveCameraEffect module.

        Args:
            strides (int, optional): The strides value. Defaults to 1.
        """
        super(RemoveCameraEffect, self).__init__()
        self.strides = strides
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the RemoveCameraEffect module.

        Returns:
            dict: The configuration dictionary.
        """
        return {'strides': self.strides}
        
    def forward(self, v_cam: torch.Tensor, cam_K: torch.Tensor, coord_K: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the RemoveCameraEffect module.

        Args:
            v_cam (torch.Tensor): The v_cam tensor.
            cam_K (torch.Tensor): The cam_K tensor.
            coord_K (torch.Tensor): The coord_K tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.einsum('bxyij, bxyj-> bxyi', self.make_Rpxy(v_cam.shape[1:3], cam_K, coord_K), v_cam)
    
    def make_Rpxy(self, shape: tuple, cam_K: torch.Tensor, coord_K: torch.Tensor) -> torch.Tensor:
        """
        Generates the RpxRpy tensor.

        Args:
            shape (tuple): The shape tuple.
            cam_K (torch.Tensor): The cam_K tensor.
            coord_K (torch.Tensor): The coord_K tensor.

        Returns:
            torch.Tensor: The RpxRpy tensor.
        """
        f = torch.stack([cam_K[:,0,0], cam_K[:,1,1]], dim=-1)
        c = cam_K[:,:2,2]

        u, v = generate_px_coordinates(shape, coord_K, self.strides)
        coords_c = torch.stack([u - c[:,0][:,None,None],
                             v - c[:,1][:,None,None]
                            ], dim=-1)

        coords_3d_with_z1 = torch.cat([coords_c / f[:,None,None], torch.ones_like(coords_c[:,:,:,:1])], dim=-1)
        z = torch.tensor([0,0,1], dtype=coords_3d_with_z1.dtype)

        axes = torch.cross(z * torch.ones_like(coords_3d_with_z1), coords_3d_with_z1)
        axes /= torch.norm(axes, dim=-1, keepdim=True) + 0.000001

        coords_3d_with_z1 = torch.isfinite(coords_3d_with_z1).all().float()
        angles = angle_between(z, coords_3d_with_z1)

        angles = torch.isfinite(angles).all().float()
        axes = torch.isfinite(axes).all().float()
        RpxRpy = make_R_from_angle_axis(angles, axes)   
        RpxRpy = torch.isfinite(RpxRpy).all().float()
        
        return RpxRpy
