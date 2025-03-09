import cv2
import os
import random
import numpy as np
from skimage import measure


def percent(per):
    # per%返回True
    rand_num = random.randint(1, 100)
    if rand_num <= per:
        return True
    else:
        return False


if __name__ == "__main__":
    mask_path = "D:\\111Project\\data\\SIRST\\masks\\"
    save_path = "D:\\111Project\\data\\SIRST\\masks_60\\"
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(mask_path):
        mask = cv2.imread(mask_path+img_name, 0)
        masked_mask = np.ones_like(mask)
        label_image = measure.label(mask)
        target_num = label_image.max()
        for index in range(1, target_num+1):
            if not percent(60):
                masked_mask -= (label_image == index).astype(np.uint8)
        mask = mask * masked_mask
        cv2.imwrite(save_path+img_name, mask)





