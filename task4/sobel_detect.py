from cv2 import cv2
import numpy as np

import os
from PIL import Image, ImageEnhance


# 我采用sobel算子的实现图像边缘检测
def sobel(img):
    row, col = img.shape
    # 初始化
    image = np.zeros((row, col))
    G_x = np.zeros((row, col))  # 横向检测图像Gx
    G_y = np.zeros((row, col))  # 纵向检测图像Gy
    # x方向的卷积核(算子)
    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # y方向的卷积核
    s_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in range(row - 2):  # 卷积核x方向平移，r-2确保移动到最右侧不越界
        for j in range(col - 2):  # 卷积核y方向平移，c-2确保移动到最右侧不越界
            G_x_sum = np.sum(img[i:i + 3, j:j + 3] * s_x)  # x方向的通过卷积核与图像卷积运算后求和
            G_y_sum = np.sum(img[i:i + 3, j:j + 3] * s_y)  # y方向的通过卷积核与图像卷积运算后求和
            G_x[i + 1, j + 1] = abs(G_x_sum)  # 绝对值
            G_y[i + 1, j + 1] = abs(G_y_sum)  # 绝对值
            image[i + 1, j + 1] = (G_x[i + 1, j + 1] ** 2 + G_y[i + 1, j + 1] ** 2) ** 0.5  # 求梯度

    return np.uint8(image)


img = cv2.imread('./data/test.jpg', cv2.IMREAD_GRAYSCALE)  # 转化为灰度图像
cv2.imshow('image', img)

out_sobel = sobel(img)
cv2.imshow('sobel_image', out_sobel)
cv2.imwrite('./data/sobel_detect.jpg', out_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()
