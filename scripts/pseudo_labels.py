import cv2
import numpy as np
import os
from skimage import measure


def get_connected_component(image, coord):
    """
    保留与给定坐标连通的图像部分，并返回处理后的图像。


    """
    # 读取图像
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 如果在给定坐标的像素值为背景（假设为0），则返回原图像


    # 为洪水填充准备mask，大小为(w+2, h+2)，此mask用于确定填充区域的范围
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    if image[coord[1], coord[0]] == 0:
        ans = np.zeros((h, w))
        ans[coord[1], coord[0]] = 255
        return ans
    # 使用洪水填充
    flood_flags = 8
    flood_mask = mask.copy()
    cv2.floodFill(image, flood_mask, coord, 255, (5,), (5,), flood_flags)

    # 使用mask创建输出图像，只保留与指定坐标连通的部分
    out_img = np.zeros_like(image)
    out_img[flood_mask[1:-1, 1:-1] == 1] = 255

    return out_img


if __name__ == "__main__":
    img_path = "D:\\111Project\\data\\SIRST\\images\\"
    mask_point_path = "D:\\111Project\\data\\SIRST\\masks_coarse\\"
    save_path = "D:\\111Project\\data\\SIRST\\masks_t09\\"
    # out_path = "D:\\111Project\\data\\ans\\SIRST\\TLLCM\\"
    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(img_path):
        img = cv2.imread(img_path+img_name, 0)
        m, n = img.shape
        ans = np.zeros((m, n))
        mask_point = cv2.imread(mask_point_path+img_name)

        # binary_image = cv2.imread(out_path+img_name, 0)
        label_image = measure.label(mask_point)
        for region in measure.regionprops(label_image, cache=False):

            centroid = [int(region.centroid[1]), int(region.centroid[0])]
            center = img[centroid[1], centroid[0]]
            _, binary_image = cv2.threshold(img, center * 0.9, 255, cv2.THRESH_BINARY)
            ans += get_connected_component(binary_image, centroid)
        cv2.imwrite(save_path+img_name, ans.astype(np.uint8))



