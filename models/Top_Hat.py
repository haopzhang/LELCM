import cv2
import numpy as np
import os


def Top_Hat(image):
    # 定义形态学操作的核（kernel），这里使用一个3x3的矩形核
    # 2. 定义一个合适大小的结构元素
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 3. 对图像进行开操作
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)

    # 4. 使用 Top-Hat 变换
    top_hat = cv2.subtract(image, opened_image)

    # 5. 对 Top-Hat 变换的结果进行二值化操作
    _, binary_image = cv2.threshold(top_hat, 30, 255, cv2.THRESH_BINARY)
    return binary_image



if __name__ == "__main__":
    imgs_path = "D:\\111Project\\data\\IRSTD-1k\\images\\"
    save_path = "D:\\111Project\\data\\ans\\IRSTD-1k\\Top-Hat\\"
    os.makedirs(save_path, exist_ok=True)
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img = cv2.imread(imgs_path+img_name)
        ans = Top_Hat(img)
        cv2.imwrite(save_path+img_name, ans)

