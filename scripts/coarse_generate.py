import cv2
import numpy as np
import os
from skimage import measure
import random


if __name__ == "__main__":
    mask_path = "D:\\111Project\\data\\IRSTD-1k\\masks\\"
    save_path = "D:\\111Project\\data\\IRSTD-1k\\masks_coarse\\"
    os.makedirs(save_path, exist_ok=True)
    mask_point_path = "D:\\111Project\\data\\IRSTD-1k\\masks_centroid\\"
    img_names = os.listdir(mask_path)
    for img_name in img_names:
        mask = cv2.imread(mask_path + img_name, 0)
        mask_point = cv2.imread(mask_point_path + img_name, 0)
        m, n = mask.shape
        ans = np.zeros((m, n))
        label_image = measure.label(mask_point)
        times = 0
        for region in measure.regionprops(label_image, cache=False):
            x, y = int(region.centroid[0]), int(region.centroid[1])
            while True:
                x_ = x + random.randint(-2, 2)
                y_ = y + random.randint(-2, 2)
                if x_ >= m or y_ >= n or x < 0 or y < 0:
                    continue
                if mask[x_, y_] == 255:
                    ans[x_, y_] = 255
                    break
                times += 1
                if times > 10000:
                    ans[x, y] = 255
                    break
        cv2.imwrite(save_path+img_name, ans)
