import cv2
import numpy as np
import torch
from scipy.signal import convolve2d
import os



def local_contrast_enhancement(image, ksize=3):
    """
    局部对比度增强
    """
    local_mean = cv2.boxFilter(image, -1, (ksize, ksize))
    enhanced = image - local_mean + 128
    return np.clip(enhanced, 0, 255).astype(np.uint8)

def background_estimation(image, ksize=15):
    """
    背景估计
    """
    return cv2.medianBlur(image, ksize)

def target_detection(enhanced, background):
    """
    从背景中减去增强图像并进行二值化
    """
    diff = cv2.subtract(enhanced, background)
    _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return binary


def TLLCM(image, cell_sizes, K=9, lamda=0.6):
    if isinstance(image, torch.Tensor):
        image = (np.squeeze(image.cpu().numpy())*255).astype(np.uint8)
    m, n = image.shape
    LCM = np.zeros([len(cell_sizes)+1, m, n])
    gaussian_core = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    conv_img = convolve2d(image, gaussian_core, mode='same', boundary='fill', fillvalue=0)
    for index in range(0, len(cell_sizes)):
        cell_size = cell_sizes[index]
        pad_img = np.pad(image, 2*cell_size, 'constant')
        pad_len = (cell_size-1)//2
        for i in range(0, m):
            for j in range(0, n):
                I_surr = []
                top_left = [i-pad_len+cell_size, j-pad_len+cell_size]
                for ii in range(0, 3):
                    for jj in range(0, 3):
                        if ii == 1 and jj == 1:
                            continue
                        region = pad_img[top_left[0]+ii*cell_size:top_left[0]+(ii+1)*cell_size, top_left[1]+jj*cell_size:top_left[1]+(jj+1)*cell_size]
                        pixels = region.flatten()
                        largest_k = np.sort(pixels)[::-1][:K]
                        average_k = np.sum(largest_k) / K

                        if abs(average_k) < 1e-8:
                            if conv_img[i, j] < 1e-8:
                                I_surr.append(0)
                            else:
                                I_surr.append(0)
                        else:
                            I_surr.append(conv_img[i, j]/average_k*conv_img[i, j]-conv_img[i, j])

                LCM[index, i, j] = min(I_surr)

        SM = np.amax(LCM, axis=0)
        thresh = lamda*np.max(SM) + (1-lamda)*np.mean(SM)
        binary_img = np.where(SM > thresh, 255, 0)
        # return binary_img
        return SM*((SM>0).astype(np.uint8))*25


def RLCM(image):
    enhanced = local_contrast_enhancement(image)
    background = background_estimation(enhanced)
    binary = target_detection(enhanced, background)
    return binary


if __name__ == "__main__":
    dataset_name = "SIRST\\"
    data_path = "D:\\111Project\\data\\"
    mask_path = data_path + dataset_name + "masks\\"
    img_path = data_path + dataset_name + "images\\"
    save_path = data_path + "ans\\" + dataset_name + "TLLCM_SM\\"
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        img = cv2.imread(img_path+img_name, 0)
        ans = TLLCM(img, cell_sizes=[5, 7, 9, 11])
        ans = np.array(ans, dtype=np.uint8)
        cv2.imwrite(save_path+img_name, ans)

