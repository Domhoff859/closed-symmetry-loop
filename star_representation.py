from utils import *
import numpy as np
from math import isclose
import torch
from typing import Tuple

def collapses_obj_to_dot_symmetry(obj: torch.Tensor, x_factor: float = 1, y_factor: float = 1, z_factor: float = 1) -> torch.Tensor:
    """
    Applies rotations around the x, y, and z axes to the input object.

    Args:
        obj (torch.Tensor): The input object.
        x_factor (float): The rotation factor around the x-axis.
        y_factor (float): The rotation factor around the y-axis.
        z_factor (float): The rotation factor around the z-axis.

    Returns:
        torch.Tensor: The object after applying the rotations.
    """
    # Create an identity matrix with the same batch shape as the input object
    R = torch.eye(3, batch_shape=(obj.size())[:-1])
    
    # Apply rotations around the x, y, and z axes to the object
    obj = change_Angle_around_Axis(R[...,0], obj, R[...,1], x_factor)
    obj = change_Angle_around_Axis(R[...,1], obj, R[...,2], y_factor)
    obj = change_Angle_around_Axis(R[...,2], obj, R[...,0], z_factor)
    
    return obj

class StarRepresentation(torch.nn.Module):
    def __init__(self, model_info: dict, **kwargs):
        """
        Initializes the StarRepresentation module.

        Args:
            model_info (dict): Information about the model.
            **kwargs: Additional keyword arguments.
        """
        super(StarRepresentation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.model_info = model_info
        
    def forward(self, po: torch.Tensor) -> torch.Tensor:
        """
        Applies the star representation to the input object.

        Args:
            po (torch.Tensor): The input object.

        Returns:
            torch.Tensor: The object after applying the star representation.
        """
        if self.model_info["symmetries_continuous"]:
            # If continuous symmetries are enabled, collapse the object to dot symmetry
            print("Starring as symmetries_continuous")
            return collapses_obj_to_dot_symmetry(po, z_factor=np.inf)

        if len(self.model_info["symmetries_discrete"]) == 0:
            # If there are no discrete symmetries, return the original object
            print("Starring is not changing anything")
            return po

        if isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            # If the z-axis is a discrete symmetry, correct the object's position and collapse to dot symmetry
            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            po = po + offset
            print("po was corrected by", offset)

            print("Starring as symmetries_discrete with z_factor=", len(self.model_info["symmetries_discrete"])+1)
            return collapses_obj_to_dot_symmetry(po, z_factor=len(self.model_info["symmetries_discrete"])+1)


        if isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            # If the y-axis is a discrete symmetry, correct the object's position and collapse to dot symmetry
            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            po = po + offset
            print("po was corrected by", offset)

            print("Starring as symmetries_discrete with y_factor=", len(self.model_info["symmetries_discrete"])+1)
            return collapses_obj_to_dot_symmetry(po, y_factor=len(self.model_info["symmetries_discrete"])+1)

        assert(False)
        
    def get_config(self) -> dict:
        """
        Returns the configuration of the StarRepresentation module.

        Returns:
            dict: The configuration of the module.
        """
        config = {'model_info': self.model_info}
        base_config = super(StarRepresentation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
