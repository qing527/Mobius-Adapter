#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
from dataset.matterport3d import Matterport3D
from dataset.pnvs import PNVS
from torch.utils.data import DataLoader

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from PIL import Image

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from models.unet_2d_condition import UNet2DConditionModel
from network.panodepth_pipeline import PanoDepthPipeline
from dataset.metrics import Evaluator
from diffusers.models.attention_processor import AttnProcessor2_0
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")
from network.my_tb_tracker import MyCustomTracker
logger = get_logger(__name__, log_level="INFO")


def log_validation(vae,text_encoder,tokenizer,unet,args,accelerator,weight_dtype,scheduler,epoch,val_loader,tb_tracker=None, global_step=0
                   ):
    
    denoise_steps = 10
    ensemble_size = 10
    processing_res = 1024
    match_input_res = False
    batch_size = 1
    color_map="Spectral"
    
    
    logger.info("Running validation ... ")
    pipeline = PanoDepthPipeline.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name_or_path, 
                                                   vae=accelerator.unwrap_model(vae),
                                                   text_encoder=accelerator.unwrap_model(text_encoder),
                                                   tokenizer=tokenizer,
                                                   unet = accelerator.unwrap_model(unet),
                                                   safety_checker=None,
                                                   scheduler = accelerator.unwrap_model(scheduler),
                                                   torch_dtype=weight_dtype)

    pipeline = pipeline.to(accelerator.device)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass  

    pbar = tqdm(val_loader)
    pbar.set_description("Validating Epoch_{}".format(epoch))
    evaluator = Evaluator(pano_h = 512, pano_w = 1024)
    evaluator.reset_eval_metrics()

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):

            # Predict depth
            pipe_out = pipeline(
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
            mask = inputs["val_mask"].to(depth_pred.device)
            depth_pred1 = evaluator.compute_eval_metrics(gt_depth, depth_pred[None, None, ...], mask, rgb_type="pano")
            
            if batch_idx < 2:
                tb_rgb = np.transpose(inputs["rgb"][0,...].detach().cpu().numpy(),(1,2,0)) 
                tb_pred_depth = depth_pred1.detach().clone().squeeze().cpu().numpy()
                tb_gt_depth = inputs["gt_depth"].detach().cpu().numpy()[0,0,...]
                tb_mask = ((tb_gt_depth > 0) & (tb_gt_depth <= 10.0) & ~np.isnan(tb_gt_depth))
                tb_tracker.add_image("pred_depth/{}".format(batch_idx), img=np.array(depth_colored), global_step=global_step, dataformats='HWC')
                tb_tracker.add_sphere_pointclouds(tag="pointclouds/{}".format(batch_idx), rgb=tb_rgb, 
                                                  pred_depth=tb_pred_depth, gt_depth=tb_gt_depth, mask=tb_mask, step=global_step)

        
        evaluator.print()
        for i, key in enumerate(evaluator.metrics.keys()):
            v = np.array(evaluator.metrics[key].avg.cpu())
            tb_tracker.add_scalar(tag=key, scalar_value=v, global_step=global_step)

        del depth_colored
        del pipeline
        torch.cuda.empty_cache()



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # 
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    # pretrained model settings 
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument( # exchange the identifier
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    # training settings
    # dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="A seed for reproducible training.")
    # data processing
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument( # --random_flip true
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # Training parameter settings
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument( # prior
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument( # the step of gradient update
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument( 
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # learning rate settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument( # 0 
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # optimizer parameters
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. "
        " If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # logging settings
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument( # default
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # checkoint settings
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5000,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="normal",
        help=(
            "noise type: normal Gaussian, pyramid Gaussian, annealed pyramid Gaussian"
        ),
    )

    args = parser.parse_args()
    # checking
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir) 
    mycustomtracker = MyCustomTracker(run_name='tb_logs', logging_dir=args.output_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=mycustomtracker,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
                                            in_channels=8,
                                            low_cpu_mem_usage=False,
                                            ignore_mismatched_sizes=True)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    unet.set_attn_processor(AttnProcessor2_0())


    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", 
                                                in_channels=8,
                                                low_cpu_mem_usage=False,
                                                ignore_mismatched_sizes=True)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    # whether using xformers /check xformers version
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
        # when save model exe save_model_hook
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    # speed up train process
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # pnvs
    train_dataset = PNVS(root_dir = "/data/liuqing/Dataset/PNVS/", list_file = "/data/liuqing/Deformable_SD/dataset/PNVS_hard.txt", 
                                         height=512, width=1024, 
                                         disable_LR_filp_augmentation=(not args.random_flip),
                                         disable_crop_augmentation=(not args.random_crop),
                                         crop_resolution=args.resolution,
                                         is_training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                       num_workers=args.dataloader_num_workers, pin_memory=True, drop_last=True)


    test_dataset = Matterport3D(root_dir = "/data/liuqing/Dataset/MP3D_unifuse/", list_file = "/data/liuqing/Deformable_SD/dataset/matterport3d_test.txt", 
                                         height=512, width=1024, 
                                         disable_color_augmentation=False, 
                                         disable_LR_filp_augmentation=False, 
                                         disable_yaw_rotation_augmentation=False,
                                         is_training=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=True, drop_last=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    # scale factor.
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        if mycustomtracker is None:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process:
        unet.eval()
        log_validation(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            args=args,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            scheduler=noise_scheduler,
            epoch=0,
            val_loader=test_dataloader,
            tb_tracker=mycustomtracker,
            global_step = 0
        )

    def pyramid_noise_like(x, discount=0.9):
        b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
        u = torch.nn.Upsample(size=(w, h), mode='bilinear')
        noise = torch.randn_like(x)
        for i in range(10):
            r = random.random()*2+2 # Rather than always going 2x, 
            w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
            noise += u(torch.randn(b, c, w, h).to(x)) * discount**i
            if w==1 or h==1: break # Lowest resolution is 1x1
        return noise/noise.std() # Scaled back to roughly unit variance
    
    def annealed_pyramid_noise_like(x, timestep, num_train_timesteps=1000, discount=0.9):
        b, c, w, h = x.shape # EDIT: w and h get over-written, rename for a different variant!
        u = torch.nn.Upsample(size=(w, h), mode='bilinear')
        noise = torch.randn_like(x)
        for i in range(10):
            r = random.random()*2+2 # Rather than always going 2x, 
            w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
            weight = (discount * timestep / num_train_timesteps)**i

            noise += u(torch.randn(b, c, w, h).to(x)) * weight.view(b, 1, 1, 1)
            if w==1 or h==1: break # Lowest resolution is 1x1
        return noise/noise.std() # Scaled back to roughly unit variance

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train() 
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # convert the images and the depths into lantent space.
                rgb = batch['rgb']
                gt_depth = batch['gt_depth']
                mask = batch["val_mask"]
                
                gt_depth_normalized = batch['gt_norm_depth'].repeat(1,3,1,1)



                # encode RGB to lantents
                h_rgb = vae.encoder(rgb.to(weight_dtype))
                moments_rgb = vae.quant_conv(h_rgb)
                mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
                rgb_latents = mean_rgb *rgb_latent_scale_factor 
                
                # encode depth to lantents
                h_depth = vae.encoder(gt_depth_normalized.to(weight_dtype))
                moments_depth = vae.quant_conv(h_depth)
                mean_depth, _ = torch.chunk(moments_depth, 2, dim=1)
                depth_latents = mean_depth * depth_latent_scale_factor


                bsz = depth_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=depth_latents.device)
                timesteps = timesteps.long()

                # add noise to the depth lantents
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(depth_latents)
                if args.noise_type == "pyramid_noise":
                    noise = pyramid_noise_like(depth_latents)
                if args.noise_type == "annealed_pyramid_noise":
                    noise = annealed_pyramid_noise_like(depth_latents, timesteps, num_train_timesteps=noise_scheduler.config.num_train_timesteps)
                noisy_depth_latents = noise_scheduler.add_noise(depth_latents, noise, timesteps)
                
                # Encode text embedding for empty prompt
                prompt = ""
                text_inputs =tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(text_encoder.device) #[1,2]
                empty_text_embed = text_encoder(text_input_ids)[0].to(weight_dtype)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon": 
                    target = noise

                elif noise_scheduler.config.prediction_type == "v_prediction": 
                    target = noise_scheduler.get_velocity(depth_latents, noise, timesteps)

                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                batch_empty_text_embed = empty_text_embed.repeat((noisy_depth_latents.shape[0], 1, 1)) 
                
                # predict the noise residual and compute the loss.
                unet_input = torch.cat([rgb_latents,noisy_depth_latents], dim=1)  
                
                # predict the noise residual
                noise_pred = unet(unet_input, 
                                  timesteps, 
                                  encoder_hidden_states=batch_empty_text_embed).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                mycustomtracker.add_scalar(tag='training loss',
                                   scalar_value=train_loss,
                                   global_step=global_step)
                # accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # validation
                        unet.eval()
                        log_validation(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            scheduler=noise_scheduler,
                            epoch=epoch,
                            val_loader=test_dataloader,
                            tb_tracker=mycustomtracker,
                            global_step = global_step
                        )
                        unet.train() 

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            # validation inference here
            unet.eval()
            log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                scheduler=noise_scheduler,
                epoch=epoch,
                val_loader=test_dataloader,
                tb_tracker=mycustomtracker,
                global_step = global_step
            )
            
            if args.use_ema:
                # Switch back to the original UNet parameters.
                ema_unet.restore(unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()