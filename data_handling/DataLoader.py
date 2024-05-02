import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import torchvision.transforms as transforms

def open_annotator(name):
    """
    Opens and reads a JSON file.

    Args:
        name (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file.
    """
    assert(Path(name).is_file())
    
    with open(name) as f:
        return json.load(f)

def load_gt_data(root_dirs, oiu):
    """
    Loads ground truth data from multiple directories.

    Args:
        root_dirs (list): A list of root directories.
        oiu (int): The object ID to filter the data.

    Returns:
        list: A list of dictionaries containing the loaded data.
    """
    found_data = []
    
    for rd in root_dirs:
        for root, sub_dirs, files in os.walk(rd):
            for sd in tqdm(sub_dirs):
                dir = f'{root}/{sd}'
                print(dir)
                scene_gt = open_annotator(f'{dir}/scene_gt.json')
                scene_gt_info = open_annotator(f'{dir}/scene_gt_info.json')
                scene_camera = open_annotator(f'{dir}/scene_camera.json')
                
                assert(len(scene_gt) == len(scene_gt_info))
                assert(len(scene_gt) == len(scene_camera))
                
                for key, gt_values in scene_gt.items():
                    for vi, v in enumerate(gt_values):
                        if v["obj_id"] == oiu and scene_gt_info[key][vi]["visib_fract"] > 0.1: 
                            
                            new_data = {}
                            new_data['root'] = dir
                            new_data['file_name'] = "{:06d}".format(int(key))
                            new_data['oi_name'] = "{:06d}".format(vi)
                            new_data['cam_R_m2c'] = np.array(v["cam_R_m2c"]).reshape((3,3))
                            new_data['cam_t_m2c'] = np.array(v["cam_t_m2c"])
                            
                            bbox_obj = scene_gt_info[key][vi]["bbox_obj"]
                            new_data['bbox_start'] = np.array(bbox_obj[:2])
                            new_data['bbox_dims'] = np.array(bbox_obj[2:])
                            
                            new_data['cam_K'] = np.array(scene_camera[key]["cam_K"]).reshape((3,3))
                            new_data['depth_scale'] = scene_camera[key]["depth_scale"]
                            
                            new_data['visib_fract'] = scene_gt_info[key][vi]["visib_fract"]
                            
                            found_data.append(new_data)
            break
    return found_data

def load_foreign_data(root_dirs, foreign_info, oiu):
    """
    Loads foreign data from multiple directories.

    Args:
        root_dirs (list): A list of root directories.
        foreign_info (str): The name of the foreign info file.
        oiu (int): The object ID to filter the data.

    Returns:
        list: A list of dictionaries containing the loaded data.
    """
    found_data = []
    
    for rd in root_dirs:
        for root, sub_dirs, files in os.walk(rd):
            for sd in tqdm(sub_dirs):
                dir = f'{root}/{sd}'
                
                scene_foreign_info = open_annotator(f'{dir}/{foreign_info}')
                scene_camera = open_annotator(f'{dir}/scene_camera.json')
                
                assert(len(scene_foreign_info) == len(scene_camera))
                
                for key, foreign_values in scene_foreign_info.items():
                    for vi, v in enumerate(foreign_values):
                        if v["obj_id"] == oiu and v["score"] > 0.0:
                            
                            new_data = {}
                            new_data['root'] = dir
                            new_data['file_name'] = "{:06d}".format(int(key))
                            new_data['oi_name'] = "{:06d}".format(vi)
                                                        
                            bbox_obj = np.array(v["bbox_obj"]).astype(float)
                            new_data['bbox_start'] = bbox_obj[:2]
                            new_data['bbox_dims'] = bbox_obj[2:] - bbox_obj[:2]
                            
                            new_data['cam_K'] = np.array(scene_camera[key]["cam_K"]).reshape((3,3))
                            new_data['depth_scale'] = scene_camera[key]["depth_scale"]
                            
                            new_data['score'] = v["score"]
                                                        
                            found_data.append(new_data)
            break
            
    return found_data


def load_data_item(datum, test_mode=False):
    """
    Loads a single data item.

    Args:
        datum (dict): The data dictionary.
        test_mode (bool, optional): Whether to load data for testing. Defaults to False.

    Returns:
        tuple: A tuple containing the loaded data.
    """
    img = np.array(Image.open(f'{datum["root"]}/rgb/{datum["file_name"]}{".png" if "primesense" in datum["root"] else ".jpg"}'))
    depthimg = np.array(Image.open(f'{datum["root"]}/depth/{datum["file_name"]}.png'), np.float32)
    depthimg *= datum["depth_scale"]

    if test_mode:
        return img, depthimg, datum["cam_K"], datum['bbox_start'], datum['bbox_dims']
    
    seg = np.array(Image.open(f'{datum["root"]}/mask_visib/{datum["file_name"]}_{datum["oi_name"]}.png'))
    return img, depthimg, seg, datum["cam_K"], datum["cam_R_m2c"], datum["cam_t_m2c"], datum['bbox_start'], datum['bbox_dims']

def extract_item(datum, xyDim, sigma=0.2, test_mode=False):
    """
    Extracts an item from the data.

    Args:
        datum (tuple): The data tuple.
        xyDim (int): The dimension of the item.
        sigma (float, optional): The sigma value for random scaling. Defaults to 0.2.
        test_mode (bool, optional): Whether to extract item for testing. Defaults to False.

    Returns:
        tuple: A tuple containing the extracted item.
    """
    if test_mode:
        img, depth, cam_K, bbs, bbd = datum
        
        scale = bbd.max() / xyDim
        new_bbs = bbs + (bbd - bbd.max()) / 2
    else:
        img, depth, seg, cam_K, R, t, bbs, bbd = datum
        
        scale_diff = np.maximum(np.random.normal(1, sigma), 0.6)
        scale = bbd.max() / xyDim * scale_diff
        new_bbs = bbs + (bbd - bbd.max()) / 2 - (scale_diff- 1) * bbd.max() / 2  + np.random.normal(0, sigma * bbd.max() / 2., 2)

    transformation = [scale, 0, new_bbs[0], 0.0, scale,  new_bbs[1], 0.0, 0.0]
    coord_K = np.stack([np.array([scale,scale]), new_bbs])
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((xyDim, xyDim)),
        transforms.ToTensor()
    ])
    
    transformed_img = transform(img)
    transformed_depth = transform(depth)
    
    if test_mode:
        return transformed_img, transformed_depth, cam_K, coord_K
    else:
        transformed_seg = transform(seg)
        return transformed_img, transformed_depth, transformed_seg, cam_K, R, t, coord_K
    
class CustomDataset(Dataset):
    def __init__(self, data_, xyDim, times=1, group_size=1, random=False, sigma=0.2, test_mode=False):
        """
        Custom dataset class.

        Args:
            data_ (list): The data list.
            xyDim (int): The dimension of the item.
            times (int, optional): The number of times to repeat the data. Defaults to 1.
            group_size (int, optional): The group size. Defaults to 1.
            random (bool, optional): Whether to use random data. Defaults to False.
            sigma (float, optional): The sigma value for random scaling. Defaults to 0.2.
            test_mode (bool, optional): Whether to use test mode. Defaults to False.
        """
        self.data = data_ * times
        self.xyDim = xyDim
        self.group_size = group_size
        self.random = random
        self.sigma = sigma
        self.test_mode = test_mode
        
    def __len__(self):
        return len(self.data) * self.group_size
    
    def __getitem__(self, idx):
        d = self.data[idx // self.group_size]
        return extract_item(load_data_item(d, test_mode=self.test_mode), self.xyDim, sigma=self.sigma, test_mode=self.test_mode)
