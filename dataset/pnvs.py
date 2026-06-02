from __future__ import print_function
import os
import cv2
import numpy as np
import random
import math
import torch
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plot

def make_coord(shape, ranges=(-1, 1), flatten=False):
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges # -1,1
        r = (v1 - v0) / (2 * n) # (max-min)/2h
        seq = v0 + r + (2 * r) * torch.arange(n).float() # min+r+2*r
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list

def normalize_depth(depth_map):
    """
    Normalize depth map using the provided formula:
    d' = ((d - d2) / (d98 - d2) - 0.5) * 2
    where d2 and d98 are the 2% and 98% percentiles of the depth map.

    Parameters:
    - depth_map: numpy array, np.uint16, input depth map

    Returns:
    - normalized_depth: numpy array, normalized depth map
    """

    # Replace NaN values with a placeholder (e.g., -1)
    depth = np.copy(depth_map)
    depth = depth.astype(np.float32) / 1000.0

    # Calculate the 2% and 98% percentiles without considering NaN values
    d2 = np.percentile(depth[depth>0], 2)
    d98 = np.percentile(depth[depth>0], 98)

    # Apply the normalization formula and replace NaN values with -1 or 1
    normalized_depth = ((depth - d2) / (d98 - d2) - 0.5) * 2
    normalized_depth = np.where(depth_map==0, 1, normalized_depth)
    np.clip(normalized_depth, -1, 1, out=normalized_depth) 

    return normalized_depth

class PNVS(data.Dataset):
    """The PNVS Dataset"""

    max_depth_meters = 10.0

    def __init__(self, root_dir, list_file, height=480, width=640, 
                 disable_LR_filp_augmentation=False, disable_crop_augmentation=False, crop_resolution=512, is_training=False):
        """
        Args:
            root_dir (string): Directory of the PNVS Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)
        self.w = width
        self.h = height
        self.max_depth_meters = 10.0

        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.crop_augmentation = not disable_crop_augmentation
        self.crop_transform = transforms.RandomCrop((crop_resolution, crop_resolution))

        self.is_training = is_training

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h))

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        if gt_depth is None:
            raise ValueError(f"[ERROR] cv2 failed to read image: {self.rgb_depth_list[idx][1]}")
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_norm_depth = normalize_depth(gt_depth)
        gt_depth = gt_depth.astype(np.float32) / 1000.0

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
            gt_norm_depth = cv2.flip(gt_norm_depth, 1)

        rgb = np.transpose(rgb,(2,0,1))
        rgb = rgb / 255.0

        inputs["rgb"] = torch.from_numpy(rgb).to(torch.float32)
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0)).to(torch.float32)
        inputs["gt_norm_depth"] = torch.from_numpy(np.expand_dims(gt_norm_depth, axis=0)).to(torch.float32)
        inputs["rgb_file"] = rgb_name

        inputs["val_mask"] = ((inputs["gt_depth"] > 0.1) & (inputs["gt_depth"] <= self.max_depth_meters))
        # print(inputs["conditioning_pixel_values"].shape)
        if self.is_training and self.crop_augmentation:
            inputs["rgb"] = self.crop_transform(inputs["rgb"])
            inputs["gt_depth"] = self.crop_transform(inputs["gt_depth"])
            inputs["gt_norm_depth"] = self.crop_transform(inputs["gt_norm_depth"])
            inputs["val_mask"] = self.crop_transform(inputs["val_mask"])

        return inputs



