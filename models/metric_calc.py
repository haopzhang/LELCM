import cv2
import torch
import os
import numpy as np
from evaluation_metrics import *
from log import *


if __name__ == "__main__":
    ans_path = "D:\\111Project\\data\\NUDT-SIRST\\masks_TLLCM_try\\"
    mask_path = "D:\\111Project\\data\\NUDT-SIRST\\masks\\"
    metric_log = Log("Top-hat")
    img_names = os.listdir(ans_path)
    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    for img_name in img_names:
        mask = cv2.imread(mask_path+img_name, 0)
        ans = cv2.imread(ans_path+img_name, 0)
        m, n = mask.shape
        if mask.shape != ans.shape:
            continue
        eval_mIoU.update(torch.tensor(ans > 0).unsqueeze(0).unsqueeze(0).cpu(), torch.tensor(mask/255.0).unsqueeze(0).unsqueeze(0))
        eval_PD_FA.update((torch.tensor(ans > 0)).cpu(), torch.tensor(mask/255.0), [m, n])
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    metric_log.log_print("pixAcc, mIoU:\t" + str(results1))
    metric_log.log_print("PD, FA:\t" + str(results2))
