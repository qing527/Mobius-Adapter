import os
import torch
import network.spherical as S360
import numpy
import numpy as np
import math
#==========================
# Depth Prediction Metrics
#==========================
def spiral_sampling(grid, percentage):
    b, c, h, w = grid.size()    
    N = torch.tensor(h*w*percentage).int().float()    
    sampling = torch.zeros_like(grid)[:, 0, :, :].unsqueeze(1)
    phi_k = torch.tensor(0.0).float()
    for k in torch.arange(N - 1):
        k = k.float() + 1.0
        h_k = -1 + 2 * (k - 1) / (N - 1)
        theta_k = torch.acos(h_k)
        phi_k = phi_k + torch.tensor(3.6).float() / torch.sqrt(N) / torch.sqrt(1 - h_k * h_k) \
            if k > 1.0 else torch.tensor(0.0).float()
        phi_k = torch.fmod(phi_k, 2 * numpy.pi)
        sampling[:, :, int(theta_k / numpy.pi * h) - 1, int(phi_k / numpy.pi / 2 * w) - 1] += 1.0
    return (sampling > 0).float()

def recover_metric_depth_bs1(pred, gt, mask0=None):
    torch_flag = False
    device = None
    dtype = None
    if type(pred).__module__ == torch.__name__:
        torch_flag = True
        device = pred.device
        dtype = pred.dtype
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    if type(mask0).__module__ == torch.__name__:
        mask0 = mask0.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask0 = mask0.squeeze()
    
    gt_mask = gt[mask0]
    pred_mask = pred[mask0]
    a, b = numpy.polyfit(pred_mask, gt_mask, deg=1)
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = numpy.mean(pred_mask)
        gt_mean = numpy.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    if torch_flag:
        pred_metric = torch.from_numpy(pred_metric).to(dtype).to(device).unsqueeze(0).unsqueeze(0)
    return pred_metric

def compute_errors(gt, pred, invalid_mask, weights, sampling, median_scale=True):
    b, _, __, ___ = gt.size()

    scale = torch.median(gt.reshape(b, -1), dim=1)[0] / torch.median(pred.reshape(b, -1), dim=1)[0]\
        if median_scale else torch.tensor(1.0).expand(b, 1, 1, 1).to(gt.device) 
    # print(scale)
    pred = pred * scale.reshape(b, 1, 1, 1)
    valid_sum = torch.sum(~invalid_mask, dim=[1, 2, 3], keepdim=True)
    
    gt[gt < 0.1] = 0.1

    pred[pred < 0.1] = 0.1
    
    gt[invalid_mask] = 0.0
    pred[invalid_mask] = 0.0
    
    error = torch.abs(gt - pred)
    error[error < 0.1] = 0

    thresh = torch.max((gt / pred), (pred / gt))
    thresh[invalid_mask | (sampling < 0.5)] = 2.0
    
    sum_dims = [1, 2, 3]
    delta_valid_sum = torch.sum(~invalid_mask & (sampling > 0), dim=[1, 2, 3], keepdim=True)
    delta1 = (thresh < 1.25   ).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta2 = (thresh < (1.25 ** 2)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()
    delta3 = (thresh < (1.25 ** 3)).float().sum(dim=sum_dims, keepdim=True).float() / delta_valid_sum.float()

    rmse = (gt - pred) ** 2
    rmse[invalid_mask] = 0.0
    rmse_w = rmse * weights
    rmse_mean = torch.sqrt(rmse_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log[invalid_mask] = 0.0
    rmse_log_w = rmse_log * weights
    rmse_log_mean = torch.sqrt(rmse_log_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    log10 = torch.abs(torch.log10(pred/gt))
    log10[invalid_mask] = 0.0
    log10_w = log10 * weights
    log10_mean = torch.sqrt(log10_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float())

    abs_rel = (torch.abs(gt - pred) / gt)
    abs_rel[invalid_mask] = 0.0
    abs_rel_w = abs_rel * weights
    abs_rel_mean = abs_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    abs_ = torch.abs(gt - pred)
    abs_[invalid_mask] = 0.0
    abs_w = abs_ * weights
    abs_mean = abs_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    sq_rel = (((gt - pred)**2) / gt)
    sq_rel[invalid_mask] = 0.0
    sq_rel_w = sq_rel * weights
    sq_rel_mean = sq_rel_w.sum(dim=sum_dims, keepdim=True) / valid_sum.float()

    return torch.mean(abs_mean), torch.mean(abs_rel_mean), torch.mean(sq_rel_mean), torch.mean(rmse_mean), torch.mean(rmse_log_mean), \
        torch.mean(log10_mean), torch.mean(delta1), torch.mean(delta2), torch.mean(delta3), pred


# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.vals = []
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {
            'val': self.val,
            'sum': self.sum,
            'count': self.count,
            'avg': self.avg
        }

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']


class Evaluator(object):

    def __init__(self, pano_h = 256, pano_w = 512, median_align=False):
        self.weights = S360.weights.theta_confidence(S360.grid.create_spherical_grid(pano_w))
        self.sampling = spiral_sampling(S360.grid.create_image_grid(pano_w, pano_h), 0.25)

        self.median_align = median_align
        # Error and Accuracy metric trackers
        self.metrics = {}
        self.metrics["err/abs_"] = AverageMeter()
        self.metrics["err/abs_rel"] = AverageMeter()
        self.metrics["err/sq_rel"] = AverageMeter()
        self.metrics["err/rms"] = AverageMeter()
        self.metrics["err/log_rms"] = AverageMeter()
        self.metrics["err/log10"] = AverageMeter()
        self.metrics["acc/a1"] = AverageMeter()
        self.metrics["acc/a2"] = AverageMeter()
        self.metrics["acc/a3"] = AverageMeter()

    def reset_eval_metrics(self):
        """
        Resets metrics used to evaluate the model
        """
        self.metrics["err/abs_"].reset()
        self.metrics["err/abs_rel"].reset()
        self.metrics["err/sq_rel"].reset()
        self.metrics["err/rms"].reset()
        self.metrics["err/log_rms"].reset()
        self.metrics["err/log10"].reset()
        self.metrics["acc/a1"].reset()
        self.metrics["acc/a2"].reset()
        self.metrics["acc/a3"].reset()

    def compute_eval_metrics(self, gt_depth, pred_depth, mask, rgb_type="pano"):
        """
        Computes metrics used to evaluate the model
        """
        N = gt_depth.shape[0]   
   
        pred_depth = recover_metric_depth_bs1(pred_depth, gt_depth, mask)
        if pred_depth.min() < 0.1:
            pred_depth = pred_depth - pred_depth.min() + 0.1
            pred_depth = recover_metric_depth_bs1(pred_depth, gt_depth, mask)
            print("pred_depth min < 0!")


        abs_, abs_rel, sq_rel, rms, rms_log, log10, a1, a2, a3, pred = \
                compute_errors(gt_depth, pred_depth, ~mask, self.weights.to(pred_depth.device), self.sampling.to(pred_depth.device), median_scale=False)

        self.metrics["err/abs_"].update(abs_, N)
        self.metrics["err/abs_rel"].update(abs_rel, N)
        self.metrics["err/sq_rel"].update(sq_rel, N)
        self.metrics["err/rms"].update(rms, N)
        self.metrics["err/log_rms"].update(rms_log, N)
        self.metrics["err/log10"].update(log10, N)
        self.metrics["acc/a1"].update(a1, N)
        self.metrics["acc/a2"].update(a2, N)
        self.metrics["acc/a3"].update(a3, N)


        return pred

    def print(self, dir=None):
        avg_metrics = []
        avg_metrics.append(self.metrics["err/abs_"].avg)
        avg_metrics.append(self.metrics["err/abs_rel"].avg)
        avg_metrics.append(self.metrics["err/sq_rel"].avg)
        avg_metrics.append(self.metrics["err/rms"].avg)
        avg_metrics.append(self.metrics["err/log_rms"].avg)
        avg_metrics.append(self.metrics["err/log10"].avg)
        avg_metrics.append(self.metrics["acc/a1"].avg)
        avg_metrics.append(self.metrics["acc/a2"].avg)
        avg_metrics.append(self.metrics["acc/a3"].avg)

        print("\n  "+ ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log", "log10", "a1", "a2", "a3"))
        print(("&  {: 8.5f} " * 9).format(*avg_metrics))

        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'a+') as f:
                print("\n  " + ("{:>9} | " * 9).format("abs_", "abs_rel", "sq_rel", "rms", "rms_log",
                                                      "log10", "a1", "a2", "a3"), file=f)
                print(("&  {: 8.5f} " * 9).format(*avg_metrics), file=f)
