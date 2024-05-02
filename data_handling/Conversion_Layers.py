import torch
from utils import generate_px_coordinates

import torch.nn as nn
import torch.nn.functional as F


def create_Dataset_conversion_layers(xDim, yDim, model_info, strides=1):
    """
    Creates a dataset conversion layers model.

    Args:
        xDim (int): The input x dimension.
        yDim (int): The input y dimension.
        model_info (dict): Information about the model.
        strides (int, optional): The stride value. Defaults to 1.

    Returns:
        nn.Module: The conversion layers model.
    """
    class ConversionLayers(nn.Module):
        def __init__(self):
            super(ConversionLayers, self).__init__()
            self.rgb = nn.Identity()
            self.depth = nn.Identity()
            self.segmentation = nn.Identity()
            self.camera_matrix = nn.Identity()
            self.coord_offset = nn.Identity()
            self.rotation_matrix = nn.Identity()
            self.translation = nn.Identity()

        def forward(self, x):
            """
            Forward pass of the conversion layers model.

            Args:
                x (dict): The input data.

            Returns:
                tuple: A tuple containing the converted RGB image, object image, validity mask, depth, and segmentation.
            """
            depth = x['depth'][:, ::strides, ::strides]
            segmentations = x['segmentation'][:, ::strides, ::strides]
            segmentations = torch.unsqueeze(segmentations > 0, dim=-1).float()

            def depth_based_cam_coords(var):
                depth, cam_K, coord_K = var
                u, v = generate_px_coordinates(depth.shape[1:3], coord_K, strides)
                scaled_coords = torch.stack([u * depth, v * depth, depth], dim=-1)
                return torch.einsum('bij,bxyj->bxyi', torch.linalg.inv(cam_K), scaled_coords)

            cam_coords = depth_based_cam_coords((depth, x['camera_matrix'], x['coord_offset']))

            def cam_to_obj(var):
                R, t, cam_coords = var
                return torch.einsum('bji,byxj->byxi', R, cam_coords - t[:, None, None])

            obj_image = cam_to_obj((x['rotation_matrix'], x['translation'], cam_coords))
            
            def obj_validity(obj_image):
                obj_mins = torch.tensor(model_info['mins'], dtype=torch.float32) * 1.1
                obj_maxs = torch.tensor(model_info['maxs'], dtype=torch.float32) * 1.1

                obj_dim_in_range = torch.logical_and(torch.less(obj_mins, obj_image), torch.less(obj_image, obj_maxs))
                obj_dim_in_range = torch.all(obj_dim_in_range, dim=-1, keepdim=True).float()
                return obj_dim_in_range

            isvalid = obj_validity(obj_image)
            isvalid = isvalid * segmentations
            obj_image = obj_image * isvalid

            segmentation = x['segmentation'][:, ::strides, ::strides]

            return x['rgb'], obj_image, isvalid, depth, segmentation

    model = ConversionLayers()
    return model