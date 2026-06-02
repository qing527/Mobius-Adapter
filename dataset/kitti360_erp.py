"""
The dataset will convert the fisheye image to Half ERP image testing. MEI model to ERP is used.
In testing, the model is assumed being trained from erp images, the Half ERP image will be directly uses for testing.
"""

import os
import numpy as np
import cv2
import torch
import random
from PIL import Image

from .dataset import BaseDataset, resize_for_input
from .erp_geometry import fisheye_mei_to_erp, cam_to_erp_patch_fast
from torchvision import transforms

class KITTI360ERPDataset(BaseDataset):
    CAM_INTRINSIC = {
        "02": torch.tensor(
            [
                [1.33632e+03, 0.000000e00, 7.16943e+02],
                [0.000000e00, 1.33578e+03, 7.05764e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        ),
        "03": torch.tensor(
            [
                [1.48543e+03, 0.000000e00, 6.98883e+02],
                [0.000000e00, 1.48494e+03, 6.98145e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    
    camera_params_02 = {
        "dataset": "kitti360",
        "camera_model": "MEI",
        "camera_name": "image_02",
        "image_width": 1400,
        "image_height": 1400,
        "xi": 2.2134047507854890e+00,
        "k1": 1.6798235660113681e-02,
        "k2": 1.6548773243373522e+00,
        "p1": 4.2223943394772046e-04,
        "p2": 4.2462134260997584e-04,
        "fx": 1.3363220825849971e+03,
        "fy": 1.3357883350012958e+03,
        "cx": 7.1694323510126321e+02,
        "cy": 7.0576498308221585e+02
    }
    
    camera_params_03 = {
        "dataset": "kitti360",
        "camera_model": "MEI",
        "camera_name": "image_03",
        "image_width": 1400,
        "image_height": 1400,
        "xi": 2.5535139132482758e+00,
        "k1": 4.9370396274089505e-02,
        "k2": 4.5068455478645308e+00,
        "p1": 1.3477698472982495e-03,
        "p2": -7.0340482615055284e-04,
        "fx": 1.4854388981875156e+03,
        "fy": 1.4849477411748708e+03,
        "cx": 6.9888316784030962e+02,
        "cy": 6.9814541887723055e+02
    }
    
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti360_val_fisheye_erp.txt"
    # test_split = "kitti360_vis.txt"

    train_split = "kitti360_train_fisheye_erp.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        erp=True,
        tgt_f=0,
        fwd_sz=(512, 512),
        cano_sz=(1400, 1400),
        load_attn_mask=False,
        visual_debug=False,
        ico_level=4,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        self.is_dense = is_dense
        self.tgt_f = tgt_f
        self.fwd_sz = fwd_sz
        # cano_sz (h, w) is the size when model is being trained. At training time, fwd_sz (h, w) should be the same as cano_sz
        self.cano_sz = cano_sz
        self.erp = erp
        self.load_attn_mask = load_attn_mask
        self.visual_debug = visual_debug

        if self.load_attn_mask:
            # Prepare the mask for fisheye images to remove the ego-car border
            self.mask_fisheye02 = (np.load(os.path.join('splits', 'kitti360', 'mask_left_fisheye.npy'))==0).astype(np.uint8)
            self.mask_fisheye03 = (np.load(os.path.join('splits', 'kitti360', 'mask_right_fisheye.npy'))==0).astype(np.uint8)
            # Convert fisheye masks to erp masks
            self.mask_fisheye02 = fisheye_mei_to_erp(self.mask_fisheye02, self.camera_params_02, self.fwd_sz)
            self.mask_fisheye03 = fisheye_mei_to_erp(self.mask_fisheye03, self.camera_params_03, self.fwd_sz)
            self.mask_fisheye02 = cv2.resize(self.mask_fisheye02, (self.fwd_sz[1], self.fwd_sz[0]), interpolation=cv2.INTER_NEAREST)
            self.mask_fisheye03 = cv2.resize(self.mask_fisheye03, (self.fwd_sz[1], self.fwd_sz[0]), interpolation=cv2.INTER_NEAREST)
        
        self.height = fwd_sz[0]
        self.width = fwd_sz[1]            

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)
        

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join("/data/liuqing/Deformable_SD/dataset/", self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None" or not os.path.exists(
                        os.path.join(self.base_path, depth_map)
                    ):
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                    
                # setup original intrinsics
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                
                if 'image_02' in img_name:
                    img_name_pair = img_name.replace('image_02', 'image_03')
                    depth_map_pair = depth_map.replace('image_02', 'image_03') if not self.benchmark else None
                elif 'image_03' in img_name:
                    img_name_pair = img_name.replace('image_03', 'image_02')
                    depth_map_pair = depth_map.replace('image_03', 'image_02') if not self.benchmark else None
                else:
                    img_name_pair = None
                    depth_map_pair = None
                
                if img_name_pair and os.path.exists(os.path.join(self.base_path, img_name_pair)):
                    img_info["image_filename_pair"] = os.path.join(self.base_path, img_name_pair)
                    if depth_map_pair and os.path.exists(os.path.join(self.base_path, depth_map_pair)):
                        img_info["annotation_filename_depth_pair"] = os.path.join(self.base_path, depth_map_pair)
                
                
                img_info["pred_scale_factor"] = 1.0
                self.dataset.append(img_info)
        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        output = {}
        image = np.asarray(
            Image.open(self.dataset[idx]["image_filename"])
        )
        output["rgb_file"] = self.dataset[idx]["image_filename"]
        depth = (
            np.asarray(
                Image.open(self.dataset[idx]["annotation_filename_depth"])
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()

        if 'image_02' in info["image_filename"]:
            info["camera_intrinsics"] = self.CAM_INTRINSIC['02'][:, :3].clone()
            info["camera_intrinsics_pair"] = self.CAM_INTRINSIC['03'][:, :3].clone()
            cam_params = self.camera_params_02
            cam_params_pair = self.camera_params_03
            is_cam02_first = True
        elif 'image_03' in info["image_filename"]:
            info["camera_intrinsics"] = self.CAM_INTRINSIC['03'][:, :3].clone()
            info["camera_intrinsics_pair"] = self.CAM_INTRINSIC['02'][:, :3].clone()
            cam_params = self.camera_params_03
            cam_params_pair = self.camera_params_02
            is_cam02_first = False
        else:
            raise ValueError(f"Unknown camera in filename: {info['image_filename']}")

        if "image_filename_pair" in info:
            image_pair = np.asarray(Image.open(info["image_filename_pair"]))
            depth_pair = (
                np.asarray(Image.open(info["annotation_filename_depth_pair"]))
                .astype(np.float32) / self.depth_scale
            ) if "annotation_filename_depth_pair" in info else None
        else:
            image_pair = None
            depth_pair = None
        
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > 0.01
                
        # Automatically calculate the erp crop size
        crop_width = int(self.cano_sz[0])
        crop_height = int(crop_width * self.fwd_sz[0] / self.fwd_sz[1])
        
        # convert to ERP
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_height, crop_width, self.cano_sz[0],  self.cano_sz[0]*2, cam_params, roll, scale_fac=None
        )

        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
        
        if image_pair is not None:
            image_pair = image_pair.astype(np.float32) / 255.0
            depth_pair = np.expand_dims(depth_pair, axis=2) if depth_pair is not None else depth_pair
            mask_valid_depth_pair = depth_pair > 0.01 if depth_pair is not None else None
            
            image_2, depth_2, _, erp_mask_2, latitude_2, longitude_2 = cam_to_erp_patch_fast(
                image_pair, depth_pair, (mask_valid_depth_pair * 1.0).astype(np.float32) if depth_pair is not None else None,
                theta, phi, crop_height, crop_width, self.cano_sz[0], self.cano_sz[0]*2, cam_params_pair, roll, scale_fac=None
            )

            lat_range = torch.tensor([
                min(float(np.min(latitude)), float(np.min(latitude_2))),
                max(float(np.max(latitude)), float(np.max(latitude_2)))
            ])
            long_range = torch.tensor([-np.pi, np.pi])  # 360度全景

        # resizing process to fwd_sz.
        to_cano_ratio = self.cano_sz[0] / image.shape[0]
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input((image * 255.).astype(np.uint8), depth, self.fwd_sz, info["camera_intrinsics"], self.cano_sz, to_cano_ratio, mask=erp_mask)
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        if not self.test_mode:
            depth /= info['pred_scale_factor']
        
        info_2 = info.copy()
        info_2["camera_intrinsics"] = info["camera_intrinsics_pair"]
        info_2["image_filename"] = info["image_filename_pair"]
        info_2["annotation_filename_depth"] = info["annotation_filename_depth_pair"]
        image_2, depth_2, pad_2, pred_scale_factor_2, attn_mask_2 = resize_for_input((image_2 * 255.).astype(np.uint8), depth_2, self.fwd_sz, info_2["camera_intrinsics"], self.cano_sz, to_cano_ratio, mask=erp_mask_2)
        info_2['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor_2
        info_2['pad'] = pad_2
        if not self.test_mode:
            depth_2 /= info_2['pred_scale_factor']

        if self.load_attn_mask:
            if 'image_02' in self.dataset[idx]["image_filename"]:
                no_border_mask = self.mask_fisheye02.astype(np.float32)
                no_border_mask_2 = self.mask_fisheye03.astype(np.float32)
            elif 'image_03' in self.dataset[idx]["image_filename"]:
                no_border_mask = self.mask_fisheye03.astype(np.float32)
                no_border_mask_2 = self.mask_fisheye02.astype(np.float32)
            else:
                no_border_mask = None
                no_border_mask_2 = None
        else:
            no_border_mask = attn_mask > 0
            no_border_mask_2 = attn_mask_2 > 0

        image, gts, info = self.transform(image=image, gts={"depth": depth, 'attn_mask': no_border_mask}, info=info)
        image_2, gts_2, info_2 = self.transform(image=image_2, gts={"depth": depth_2, 'attn_mask': no_border_mask_2}, info=info_2)
        
        if self.visual_debug:
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(2, 2, 2)
            plt.imshow(gts["gt"].squeeze())
            plt.title("Ground Truth")
            if self.erp:
                plt.subplot(2, 2, 3)
                plt.imshow(gts["attn_mask"].squeeze())
                plt.title("Attn Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["mask"].squeeze())
            plt.title("Valid Depth Mask")
            plt.show()
        
        image = torch.cat([image, image_2], dim=2)
        gts["gt"] = torch.cat([gts["gt"], gts_2["gt"]], dim=2)
        gts["mask"] = torch.cat([gts["mask"], gts_2["mask"]], dim=2)
        gts["attn_mask"] = torch.cat([gts["attn_mask"], gts_2["attn_mask"] ], dim=2)

        erp_vis = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        erp_vis = erp_vis * std + mean
        erp_vis = erp_vis.clip(0,1)

        output["rgb"] = torch.from_numpy(erp_vis).permute(2, 0, 1).to(torch.float32)
        output["normalized_rgb"] = image
        output["gt_depth"] = gts["gt"]
        output["val_mask"] = gts["mask"].bool()
        
        return output



    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = int(image.shape[0] - self.height), int(
            (image.shape[1] - self.width) / 2
        )
        height_end, width_end = height_start + self.height, width_start + self.width
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            if depth is not None:
                height_start, width_start = int(depth.shape[0] - self.height), int(
                    (depth.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                depth = depth[height_start:height_end, width_start:width_end]
                mask = depth > self.min_depth
                if self.test_mode:
                    mask = np.logical_and(mask, depth < self.max_depth)
                    # mask = self.eval_mask(mask)
                mask = mask.astype(np.uint8)
                new_gts["gt"] = depth
                new_gts["mask"] = mask
                
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                height_start, width_start = int(attn_mask.shape[0] - self.height), int(
                    (attn_mask.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask
        return image, new_gts, info
