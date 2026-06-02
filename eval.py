# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import os
from glob import glob
import logging
import re
import matplotlib.pyplot as plot
import matplotlib
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import diffusers
from dataset.matterport3d import Matterport3D
from dataset.stanford2d3d import Stanford2D3D
from dataset.pano_loader.pano_loader import Pano3D
from dataset.kitti360_erp import KITTI360ERPDataset

from torch.utils.data import DataLoader
from network.panodepth_pipeline import PanoDepthPipeline
from network.panocontroldepth_pipeline import PanoControlDepthPipeline
from utils.seed_all import seed_all
from dataset.metrics import Evaluator
from diffusers.models.attention_processor import AttnProcessor2_0
from models.unet_2d_condition import UNet2DConditionModel
from models.MobiusAdapter import ControlNetModel
from dataset.pnvs import PNVS

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Bingxin/Marigold",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        # required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--test_name",
        type=str,
        default='matterport3d',
        help="test_name.",
    )

    parser.add_argument(
        "--test_list_file",
        type=str,
        # required=True,
        help="test_list_file.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )

    # Pano3D
    parser.add_argument("--pano3d_root", type=str, help="Path to the root folder containing the Pano3D extracted data.")
    parser.add_argument("--pano3d_part", type=str, help="The Pano3D subset to load.")
    parser.add_argument("--pano3d_split", type=str, help="The Pano3D split corresponding to the selected subset that will be loaded.")
    parser.add_argument('--pano3d_types', default=['rgb', 'gt_depth', 'val_mask'], nargs='+',
            choices=[
                'rgb', 'gt_depth', 'val_mask', 'normal', 'semantic', 'structure', 'layout',
                'color_up', 'depth_up', 'normal_up', 'semantic_up', 'structure_up', 'layout_up'
                'color_down', 'depth_down', 'normal_down', 'semantic_down', 'structure_down', 'layout_down'
                'color_left', 'depth_left', 'normal_left', 'semantic_left', 'structure_left', 'layout_left'
                'color_right', 'depth_right', 'normal_right', 'semantic_right', 'structure_right', 'layout_right'
            ],
            help='The Pano3D data types that will be loaded, one of [color, depth, normal, semantic, structure, layout], potentially suffixed with a stereo placement from [up, down, left, right].'
        )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size

    # -------------------- Preparation --------------------
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # Output directories
    if output_dir is not None:
        output_dir_color = os.path.join(output_dir, "depth_colored")
        output_dir_gt = os.path.join(output_dir, "depth_gt")
        output_dir_npy = os.path.join(output_dir, "depth_npy")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_color, exist_ok=True)
        os.makedirs(output_dir_gt, exist_ok=True)
        os.makedirs(output_dir_npy, exist_ok=True)
        logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    if args.test_name == 'matterport3d':
        test_dataset = Matterport3D(root_dir = args.input_rgb_dir, list_file = args.test_list_file, 
                                         height=512, width=1024, 
                                         disable_LR_filp_augmentation=True,
                                         is_training=False)
    elif args.test_name == 'stanford2d3d':
        test_dataset = Stanford2D3D(root_dir = args.input_rgb_dir, list_file = args.test_list_file, 
                                         height=512, width=1024, 
                                         disable_color_augmentation=False, 
                                         disable_LR_filp_augmentation=False, 
                                         disable_yaw_rotation_augmentation=False,
                                         is_training=False)
    elif args.test_name == 'shanghai':
        test_dataset = Shanghai(root_dir = args.input_rgb_dir, list_file = args.test_list_file, 
                                         height=512, width=1024, 
                                         disable_color_augmentation=False, 
                                         disable_LR_filp_augmentation=False, 
                                         disable_yaw_rotation_augmentation=False,
                                         is_training=False)
    elif args.test_name == 'pano3d':
        test_dataset = Pano3D(root=args.pano3d_root,
                                part=args.pano3d_part,
                                split=args.pano3d_split,
                                types=args.pano3d_types)

    elif args.test_name == 'kitti360':
        test_dataset = KITTI360ERPDataset(
                    test_mode=True,
                    base_path='/data/liuqing/Dataset/kitti360',
                    crop="",
                    tgt_f=0,
                    undistort_f=0,
                    fwd_sz=[512,512],
                    cano_sz=[1400,1400],
                    erp=True)
        
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=True, drop_last=True)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32


    epoch = re.findall(r'\d+', checkpoint_path)[0]
    unet = UNet2DConditionModel.from_pretrained("/data/liuqing/MobiusAdapter/train_outputs/stage1", subfolder="unet")    
    controlnet = ControlNetModel(
            in_channels=4,
            conditioning_channels=1,
            )
    
    loaded_params = torch.load(os.path.join(checkpoint_path, 'stage2.pth'))
    controlnet.load_state_dict(loaded_params, strict=False)
    pipe = PanoControlDepthPipeline.from_pretrained(args.pretrained_model_name_or_path, unet=unet, controlnet=controlnet, torch_dtype=dtype)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers

    pipe = pipe.to(device)

    pbar = tqdm(test_dataloader)
    pbar.set_description("Validating Epoch_{}".format(epoch))
    evaluator = Evaluator(pano_h=512, pano_w=1024)
    evaluator.reset_eval_metrics()

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        for batch_idx, inputs in enumerate(pbar):
            # Predict depth
            mask = inputs["val_mask"]
            pipe_out = pipe(
                    rgb = inputs['rgb'][0],
                    val_mask = None,
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    batch_size=batch_size,
                    color_map=color_map,
                    show_progress_bar=True,
                )
            depth_pred: torch.tensor = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored
            gt_depth = inputs["gt_depth"].to(depth_pred.device)
            mask = mask.to(depth_pred.device)
   
            depth_pred1 = evaluator.compute_eval_metrics(gt_depth, depth_pred[None, None, ...], mask, rgb_type="pano")

            if output_dir is not None:
                tb_pred_depth = depth_pred1.detach().clone().squeeze().cpu().numpy()
                rgb_name_base = os.path.splitext(os.path.basename(inputs["rgb_file"][0]))[0]
                pred_name_base = rgb_name_base + "_pred"

                if args.test_name == 'pano3d':
                    scene_id = os.path.basename(os.path.dirname(inputs["rgb_file"][0]))
                    
                    save_dir_npy = os.path.join(output_dir_npy, scene_id)
                    save_dir_color = os.path.join(output_dir_color, scene_id)
                    save_dir_gt = os.path.join(output_dir_gt, scene_id)
                else:
                    save_dir_npy = output_dir_npy
                    save_dir_color = output_dir_color
                    save_dir_gt = output_dir_gt
                
                os.makedirs(save_dir_npy, exist_ok=True)
                os.makedirs(save_dir_color, exist_ok=True)
                os.makedirs(save_dir_gt, exist_ok=True)

                npy_save_path = os.path.join(save_dir_npy, f"{pred_name_base}.npy")
                if os.path.exists(npy_save_path):
                    logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                np.save(npy_save_path, tb_pred_depth)
                
                abs_diff = torch.abs(depth_pred1 - gt_depth)

                diff_map = abs_diff.squeeze().cpu().numpy()
                mask_np = mask.squeeze().cpu().numpy()

                diff_map[~mask_np] = 0 
                error_vmax = 1.0
                error_map_save_path = os.path.join(
                    save_dir_color, f"{pred_name_base}_error.png"
                )

                plot.imsave(error_map_save_path, 
                        diff_map, 
                        cmap='viridis',
                        vmin=0.01,
                        vmax=error_vmax)
                
                # Colorize
                colored_save_path = os.path.join(
                    save_dir_color, f"{pred_name_base}_colored.png"
                )
                if os.path.exists(colored_save_path):
                    logging.warning(
                        f"Existing file: '{colored_save_path}' will be overwritten"
                    )
                depth_colored.save(colored_save_path)

                # save ground truth
                gt_name_base = rgb_name_base + "_gt"
                gt_colored_save_path = os.path.join(
                    save_dir_gt, f"{gt_name_base}_colored.png"
                )
                if os.path.exists(gt_colored_save_path):
                    logging.warning(
                        f"Existing file: '{gt_colored_save_path}' will be overwritten"
                    )
                gt_color = gt_depth.detach().cpu().numpy()[0,0,:,:]
                plot.imsave(gt_colored_save_path, gt_color, cmap=color_map, vmin = gt_color.min(), vmax = gt_color.max())
            evaluator.print(dir=output_dir)
    evaluator.print(dir=output_dir)