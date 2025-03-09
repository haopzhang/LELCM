import numpy as np
from scipy.signal import convolve2d
from skimage import measure
import torch.nn.functional as F
import torch


def find_n_mean(image, n):
    # 返回前n最大值均值
    if n < 1:
        n = 1
    else:
        n = int(n)
    pixels = image.flatten()
    return np.mean(np.sort(pixels)[::-1][:n])


def label_update(output, mask, lamda=0.5):
    # 输出图像， 待更新soft mask
    # [m, n] cpu
    m, n = mask.shape
    label_image = measure.label((mask > 0.5).astype(np.uint8))
    ans = np.array(mask)
    for region in measure.regionprops(label_image, cache=False):
        cur_target_mask = np.zeros((m, n))
        cur_target_mask[int(region.centroid[0]), int(region.centroid[1])] = 1
        x1, y1, x2, y2 = region.bbox
        core_len = max(x2-x1, y2-y1)*2+1
        contrast_region = (convolve2d(cur_target_mask, np.ones((core_len, core_len)), boundary='fill',
                                      mode='same') > 0).astype(np.uint8)
        target_8connect = (convolve2d(((mask>0.5).astype(np.uint8)), np.ones((3, 3)), boundary='fill',
                                      mode='same') > 0).astype(np.uint8)
        # 减最小值
        temp = output*contrast_region
        temp -= np.partition(temp.flatten(), 1)[1]
        I_core = find_n_mean((mask > 0.5).astype(np.uint8)*output*contrast_region, region.area//2)
        I_surr = find_n_mean((mask < 0.5).astype(np.uint8)*output*contrast_region, 2*region.area)
        thresh = np.max(cur_target_mask*output)*(lamda*I_surr/I_core)
        thresh_output_mask = ((output*contrast_region > thresh).astype(np.uint8))
        # 8邻域判别
        label_thresh = measure.label(thresh_output_mask > 0)
        if label_thresh.max() > 1:
            for index in range(1, label_thresh.max()+1):
                this_mask = thresh_output_mask*((label_thresh == index).astype(np.uint8))
                if (this_mask*target_8connect).sum() == 0:
                    thresh_output_mask -= this_mask
        ans = (ans*thresh_output_mask+output*thresh_output_mask)/2+ans*(1-thresh_output_mask)
    ans += (mask > 0.5).astype(np.uint8)

    return np.minimum(ans, 1.0)






