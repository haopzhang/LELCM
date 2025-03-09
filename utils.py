import numpy as np
import random
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FocalLoss(nn.Module):
    """focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """
    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                preds,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                loss_reg = self.loss_weight * focal_loss(
                    pred,
                    target,
                    alpha=self.alpha,
                    gamma=self.gamma)
                loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
                loss_total = loss_total + loss_reg
            return loss_total / len(preds)
        else:
            pred = preds
            loss_reg = self.loss_weight * focal_loss(
                pred,
                target,
                alpha=self.alpha,
                gamma=self.gamma)
            loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
            loss_total = loss_reg
            return loss_total
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
def focal_loss(pred, target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>'

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = target
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


def loss_softIoU(output, gt_mask):
    cal_loss = FocalLoss()
    target_mask, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
    ans = cal_loss(output, target_mask)
    return ans


def Normalize(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalize(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    # get image mu and sigma
    with open(dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
        train_list = f.read().splitlines()
    with open(dataset_dir+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
        test_list = f.read().splitlines()
    img_list = train_list + test_list
    img_dir = dataset_dir + '/images/'
    mean_list = []
    std_list = []
    for img_pth in img_list:
        img = Image.open(img_dir + img_pth).convert('I')
        img = np.array(img, dtype=np.float32)
        mean_list.append(img.mean())
        std_list.append(img.std())
    img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg


def random_crop(img, mask, patch_size):
    # HR: N*H*W
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def calculate_interclass_variance(hist):
    total_pixels = np.sum(hist)
    w0 = 0
    total_sum = np.dot(np.arange(256).T, hist)
    best_threshold = 0
    max_variance = 0

    for i in range(256):
        w0 += hist[i]
        w1 = total_pixels - w0
        if w1 == 0:
            break
        mean0 = np.sum(np.arange(i + 1) * hist[:i + 1]) / w0
        mean1 = (total_sum - np.sum(np.arange(i + 1) * hist[:i + 1])) / w1
        variance = w0 * w1 * ((mean0 - mean1) ** 2)
        if variance > max_variance:
            max_variance = variance
            best_threshold = i

    return best_threshold
import numpy as np
import random
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FocalLoss(nn.Module):
    """focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.

    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """
    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                preds,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                loss_reg = self.loss_weight * focal_loss(
                    pred,
                    target,
                    alpha=self.alpha,
                    gamma=self.gamma)
                loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
                loss_total = loss_total + loss_reg
            return loss_total / len(preds)
        else:
            pred = preds
            loss_reg = self.loss_weight * focal_loss(
                pred,
                target,
                alpha=self.alpha,
                gamma=self.gamma)
            loss_reg = weight_reduce_loss(loss_reg, weight, reduction, avg_factor)
            loss_total = loss_reg
            return loss_total
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
def focal_loss(pred, target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>'

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = target
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    return pos_loss + neg_loss


def loss_softIoU(output, gt_mask):
    cal_loss = FocalLoss()
    target_mask, avg_factor = gt_mask, max(1, (gt_mask.eq(1)).sum())
    ans = cal_loss(output, target_mask)
    return ans


def Normalize(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']


def Denormalize(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def get_img_norm_cfg(dataset_name, dataset_dir):
    # get image mu and sigma
    with open(dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
        train_list = f.read().splitlines()
    with open(dataset_dir+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
        test_list = f.read().splitlines()
    img_list = train_list + test_list
    img_dir = dataset_dir + '/images/'
    mean_list = []
    std_list = []
    for img_pth in img_list:
        img = Image.open(img_dir + img_pth).convert('I')
        img = np.array(img, dtype=np.float32)
        mean_list.append(img.mean())
        std_list.append(img.std())
    img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg


def random_crop(img, mask, patch_size):
    # HR: N*H*W
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h), (0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size

    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def calculate_interclass_variance(hist):
    total_pixels = np.sum(hist)
    w0 = 0
    total_sum = np.dot(np.arange(256).T, hist)
    best_threshold = 0
    max_variance = 0

    for i in range(256):
        w0 += hist[i]
        w1 = total_pixels - w0
        if w1 == 0:
            break
        mean0 = np.sum(np.arange(i + 1) * hist[:i + 1]) / w0
        mean1 = (total_sum - np.sum(np.arange(i + 1) * hist[:i + 1])) / w1
        variance = w0 * w1 * ((mean0 - mean1) ** 2)
        if variance > max_variance:
            max_variance = variance
            best_threshold = i

    return best_threshold
