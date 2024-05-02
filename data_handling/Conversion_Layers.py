import torch
from utils import generate_px_coordinates

import torch.nn as nn
import torch.nn.functional as F


class ConversionLayers(nn.Module):
    def __init__(self, xDim, yDim, model_info, strides=1):
        """
        Initializes the ConversionLayers module.

        Args:
            xDim (int): The width of the input data.
            yDim (int): The height of the input data.
            model_info (dict): Information about the model.
            strides (int, optional): The stride value for downsampling. Defaults to 1.
        """
        super(ConversionLayers, self).__init__()
        self.xDim = xDim
        self.yDim = yDim
        self.model_info = model_info
        self.strides = strides

    def forward(self, x):
        """
        Forward pass of the conversion layers model.

        Args:
            x (dict): The input data.

        Returns:
            tuple: A tuple containing the converted RGB image, object image, validity mask, depth, and segmentation.
        """
        self.check_dimensions(x)
        
        depth = x['depth'][:, ::self.strides, ::self.strides]
        segmentations = x['segmentation'][:, ::self.strides, ::self.strides]
        segmentations = torch.unsqueeze(segmentations > 0, dim=-1).float()

        def depth_based_cam_coords(var):
            """
            Calculates camera coordinates based on depth.

            Args:
                var (tuple): A tuple containing depth, camera matrix, and coordinate offset.

            Returns:
                torch.Tensor: Camera coordinates.
            """
            depth, cam_K, coord_K = var
            u, v = generate_px_coordinates(depth.shape[1:3], coord_K, self.strides)
            scaled_coords = torch.stack([u * depth, v * depth, depth], dim=-1)
            return torch.einsum('bij,bxyj->bxyi', torch.linalg.inv(cam_K), scaled_coords)

        cam_coords = depth_based_cam_coords((depth, x['camera_matrix'], x['coord_offset']))

        def cam_to_obj(var):
            """
            Converts camera coordinates to object coordinates.

            Args:
                var (tuple): A tuple containing rotation matrix, translation, and camera coordinates.

            Returns:
                torch.Tensor: Object image.
            """
            R, t, cam_coords = var
            return torch.einsum('bji,byxj->byxi', R, cam_coords - t[:, None, None])

        obj_image = cam_to_obj((x['rotation_matrix'], x['translation'], cam_coords))
        
        def obj_validity(obj_image):
            """
            Checks the validity of the object image.

            Args:
                obj_image (torch.Tensor): The object image.

            Returns:
                torch.Tensor: Validity mask.
            """
            obj_mins = torch.tensor(self.model_info['mins'], dtype=torch.float32) * 1.1
            obj_maxs = torch.tensor(self.model_info['maxs'], dtype=torch.float32) * 1.1

            obj_dim_in_range = torch.logical_and(torch.less(obj_mins, obj_image), torch.less(obj_image, obj_maxs))
            obj_dim_in_range = torch.all(obj_dim_in_range, dim=-1, keepdim=True).float()
            return obj_dim_in_range

        isvalid = obj_validity(obj_image)
        isvalid = isvalid * segmentations
        obj_image = obj_image * isvalid

        segmentation = x['segmentation'][:, ::self.strides, ::self.strides]

        return x, obj_image, isvalid, depth, segmentation
    
    def check_dimensions(self, x):
        """
        Checks the dimensions of the input data.
        
        Args:
            x (dict): The input data.
        """
        assert x['rgb'].shape == (self.yDim, self.xDim, 3,)
        assert x['depth'].shape == (self.yDim, self.xDim,)
        assert x['segmentation'].shape == (self.yDim, self.xDim,)
        assert x['segmentation'].dtype == torch.int32
        assert x['camera_matrix'].shape == (3, 3,)
        assert x['coord_offset'].shape == (2,2,)
        assert x['rotation_matrix'].shape == (3, 3,)
        assert x['translation'].shape == (3,)
