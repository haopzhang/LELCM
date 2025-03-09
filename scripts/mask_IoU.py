import numpy as np


def IoU(img, mask):
    # 0 255 np
    pixel_and = np.sum(((img > 0) & (mask > 0)).astype(np.uint8))
    pixel_or = np.sum(((img > 0) | (mask > 0)).astype(np.uint8))
    return pixel_and/pixel_or


def Pd(img, mask):
    m, n = mask.shape
    pix_and = np.sum(((img > 0) & (mask > 0)).astype(np.uint8))
    sum_mask = np.sum((mask > 0).astype(np.uint8))
    return pix_and/sum_mask


def Fa(img, mask):
    m, n = mask.shape
    pix_f = np.sum(((img > 25) & (mask == 0)).astype(np.uint8))
    sum_mask = np.sum((mask > 0).astype(np.uint8))
    return pix_f, sum_mask
