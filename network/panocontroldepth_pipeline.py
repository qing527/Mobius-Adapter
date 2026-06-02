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
from typing import Dict, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.controlnet_copy_mc_my1 import ControlNetModel

from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps
from utils.ensemble import ensemble_depths

def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class PanoControlDepthPipelineOutput(BaseOutput):
    """
    Output class for pano monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: torch.Tensor
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class PanoControlDepthPipeline(DiffusionPipeline):
    """
    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None
        self.has_calculated_flops = False


    @torch.no_grad()
    def __call__(
        self,
        rgb: torch.tensor,
        val_mask: torch.tensor, 
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        batch_size: int = 1,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> PanoControlDepthPipelineOutput:


        device = self.device
        rgb = rgb.to(device)
        rgb.to(self.dtype)

        assert rgb.min() >= 0.0 and rgb.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb] * ensemble_size)
        # exit()
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=batch_size, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, torch.min(depth_pred), torch.max(depth_pred), cmap=color_map, valid_mask=val_mask
        ).squeeze()

        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        return PanoControlDepthPipelineOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self, rgb_in: torch.Tensor, num_inference_steps: int, show_pbar: bool
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = rgb_in.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device) 
        timesteps = self.scheduler.timesteps

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                rgb_in,
                t,
                encoder_hidden_states=batch_empty_text_embed,
                controlnet_cond=rgb_in.to(self.controlnet.device),
                return_dict=False,
            )

            # predict the noise residual
            outputs = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            )
            
            noise_pred = outputs.sample

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample
        
        torch.cuda.empty_cache()
        depth = self.decode_depth(depth_latent)
        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor

        depth_latent = depth_latent.to(dtype=self.vae.dtype)
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean