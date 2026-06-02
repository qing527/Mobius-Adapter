from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter
from accelerate.tracking import GeneralTracker, on_main_process
from typing import Union
import os
import numpy as np

def get_uni_sphere_xyz(H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    return sphere_xyz

def recover_metric_depth_bs1(pred, gt, mask0=None):
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask0 = mask0.squeeze()
    
    gt_mask = gt[mask0]
    pred_mask = pred[mask0]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    # print(a, b)
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    return pred_metric

class MyCustomTracker(GeneralTracker):
    """
    my custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike],
                 **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)

    @on_main_process
    def add_image(self, tag, img, **kwargs):
        self.writer.add_image(tag=tag, img_tensor=img, **kwargs)
    
    @on_main_process
    def add_sphere_pointclouds(self, tag, rgb, pred_depth, gt_depth, mask, **kwargs):
        # rgb (512, 1024, 3) uint8 <class 'numpy.ndarray'>
        crop_ratio = 80.0 / 512.0
        pred_depth = pred_depth - pred_depth.min() + 0.1
        pred_depth = recover_metric_depth_bs1(pred_depth, gt_depth, mask)
        H, W = rgb.shape[:2]
        xyz = pred_depth[..., None] * get_uni_sphere_xyz(H, W)
        xyzrgb = np.concatenate([xyz, rgb], 2)
        crop = int(H * crop_ratio)
        xyzrgb = xyzrgb[crop:-crop]
        xyzrgb = xyzrgb.reshape(-1, 6)
        xyzrgb = xyzrgb[xyzrgb[:,2] <= 0.6]

        self.writer.add_3d(
            tag, data={
                            "vertex_positions": xyzrgb[:, :3],
                            "vertex_colors": xyzrgb[:, 3:]
                            }, 
            **kwargs)