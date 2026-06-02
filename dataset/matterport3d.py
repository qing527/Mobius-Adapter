from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms


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
    depth = depth.astype(np.float32) / 4000.0
    # print(tmp_mask.shape, depth.shape)    # (480, 640) (480, 640)

    # Calculate the 2% and 98% percentiles without considering NaN values
    d2 = np.percentile(depth[depth>0], 2)
    d98 = np.percentile(depth[depth>0], 98)

    # Apply the normalization formula and replace NaN values with -1 or 1
    normalized_depth = ((depth - d2) / (d98 - d2) - 0.5) * 2
    normalized_depth = np.where(depth_map==0, -1, normalized_depth)
    np.clip(normalized_depth, -1, 1, out=normalized_depth) 

    return normalized_depth
class Matterport3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
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

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training


        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        if os.path.isabs(self.rgb_depth_list[idx][0]):
            rgb_name = self.rgb_depth_list[idx][0]
        else:
            rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_norm_depth = normalize_depth(gt_depth)

        gt_depth = gt_depth.astype(np.float32) / 4000.0
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1


        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)
            gt_norm_depth = np.roll(gt_norm_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)
            gt_norm_depth = cv2.flip(gt_norm_depth, 1)

        rgb = np.transpose(rgb,(2,0,1))
        rgb = rgb / 255.0
        inputs["rgb"] = torch.from_numpy(rgb).to(torch.float32)

        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))
        inputs["rgb_file"] = rgb_name
        inputs["gt_norm_depth"] = torch.from_numpy(np.expand_dims(gt_norm_depth, axis=0)).to(torch.float32)

        return inputs



