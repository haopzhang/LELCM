import cv2
import numpy as np
import os
from scripts.mask_IoU import IoU
import torch.nn.functional as F
import torch
from utils import *
from skimage import measure

def thresh(image, threshold):
    m, n = image.shape
    img_thresh = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            if image[i, j] > threshold*255:
                img_thresh[i, j] = 255
    return img_thresh


def thresh_percent(image, thresh_per):
    m, n = image.shape
    img_pixels = np.zeros(256)
    ans = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            img_pixels[image[i, j]] += 1
    sum_pixels = 0
    for i in range(0, 256):
        sum_pixels += img_pixels[i]
        if sum_pixels/(m*n) > thresh_per:
            return i/255.0


def thresh_percent_region(image, thresh_per):
    m, n = image.shape
    img_pixels = np.zeros(256)
    ans = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            img_pixels[image[i, j]] += 1
    img_pixels[0] = 0
    pixel_nums = sum(img_pixels)
    sum_pixels = 0
    for i in range(0, 256):
        sum_pixels += img_pixels[i]
        if sum_pixels/(pixel_nums) > thresh_per:
            return i/255.0


def thresh_OTSU(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return calculate_interclass_variance(histogram)


if __name__ == "__main__":
    imgs_path = "D:/111Project/data/ans/IRSTD-1k/IPI_orig/"
    masks_path = "D:/111Project/data/SIRST/masks/"
    mask_point_path = "D:/111Project/data/SIRST/masks_centroid/"
    save_path = "D:/111Project/data/ans/IRSTD-1k/IPI/"
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img = cv2.imread(imgs_path+img_name, 0)
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(save_path+img_name, th2)



