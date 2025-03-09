import numpy as np
import cv2
import os


def max_mean_filter(image, window_size):
    """Max-Mean滤波"""
    padded_image = np.pad(image, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                          'reflect')
    result_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            max_val = np.max(window)
            mean_val = np.mean(window)
            result_image[i, j] = max_val - mean_val

    return result_image


def max_median_filter(image, window_size):
    """Max-Median滤波"""
    padded_image = np.pad(image, ((window_size // 2, window_size // 2), (window_size // 2, window_size // 2)),
                          'reflect')
    result_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            max_val = np.max(window)
            median_val = np.median(window)
            result_image[i, j] = max_val - median_val

    return result_image


if __name__ == "__main__":
    '''
    
    
    for dataset_name in ['NUDT-SIRST\\', 'IRSTD-1k\\', 'SIRST\\']:
        data_path = "D:\\111Project\\data\\"
        mask_path = data_path + dataset_name + "masks\\"
        img_path = data_path + dataset_name + "images\\"
        save_path = data_path + "ans\\" + dataset_name + "Max_Mean\\"
        os.makedirs(save_path, exist_ok=True)
        img_names = os.listdir(img_path)
        for img_name in img_names:
            img = cv2.imread(img_path + img_name, 0)
            ans = max_mean_filter(img, window_size=5)

            ret, binary_image = cv2.threshold(ans, 0.5 * np.max(ans), 255, cv2.THRESH_BINARY)

            cv2.imwrite(save_path + img_name, binary_image)
    '''
    for dataset_name in ['SIRST\\']:
        data_path = "D:\\111Project\\data\\"
        mask_path = data_path + dataset_name + "masks\\"
        img_path = data_path + dataset_name + "images\\"
        save_path = data_path + "ans\\" + dataset_name + "Max_Median\\"
        os.makedirs(save_path, exist_ok=True)
        img_names = os.listdir(img_path)
        for img_name in img_names:
            img = cv2.imread(img_path + img_name, 0)
            ans = max_median_filter(img, window_size=5)

            ret, binary_image = cv2.threshold(ans, 0.5 * np.max(ans), 255, cv2.THRESH_BINARY)

            cv2.imwrite(save_path + img_name, binary_image)


