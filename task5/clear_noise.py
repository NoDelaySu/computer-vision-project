import numpy as np

import random

import cv2

def blur_avg_img(img,blur_size=3):
    # 均值滤波
    return cv2.blur(img, (blur_size, blur_size))

def blur_boxfilter_img(img,blur_size=3):
    # 方框滤波
    # cv2.boxFilter(原始图像, 目标图像深度, 核大小, normalize属性)
    # 其中，目标图像深度是int类型，通常用“-1”表示与原始图像一致；核大小主要包括（3，3）和（5，5）
    return cv2.boxFilter(img, -1, (blur_size,blur_size), normalize=1)

def blur_gaussian_img(img,blur_size=5):
    # 高斯滤波
    # dst = cv2.GaussianBlur(src, ksize, sigmaX)
    # 其中，src表示原始图像，ksize表示核大小，sigmaX表示X方向方差。注意，核大小（N, N）必须是奇数，X方向方差主要控制权重。
    return cv2.GaussianBlur(img, (blur_size, blur_size),sigmaX=-0.8)

def blur_median_img(img,blur_size=3):
    # 中值滤波
    # OpenCV主要调用medianBlur()函数实现中值滤波。图像平滑里中值滤波的效果最好。
    # dst = cv2.medianBlur(src, ksize)
    # 其中，src表示源文件，ksize表示核大小。核必须是大于1的奇数，如3、5、7等。
    return cv2.medianBlur(img, blur_size)


if __name__ == "__main__":

    # Read image
    sp_noise_img = cv2.imread("./data/sp_noise_img.jpg")
    gauss_noise_img = cv2.imread("./data/gauss_noise_img.jpg")
    # 中值滤波
    median_sp_noise = blur_median_img(sp_noise_img)
    median_gauss_noise = blur_median_img(gauss_noise_img)
    # 高斯滤波
    gaussian_sp_noise = blur_gaussian_img(sp_noise_img)
    gaussian_gauss_noise = blur_gaussian_img(gauss_noise_img)
    # 均值滤波
    avg_sp_noise = blur_avg_img(sp_noise_img)
    avg_gauss_noise = blur_avg_img(gauss_noise_img)
    # 方框滤波
    boxfilter_sp_noise = blur_boxfilter_img(sp_noise_img)
    boxfilter_gauss_noise = blur_boxfilter_img(gauss_noise_img)
    # 显示原始图像
    # cv2.imshow('sp_noise_img', sp_noise_img)
    # cv2.imshow('gauss_noise_img', gauss_noise_img)
    # 中值滤波效果
    cv2.imshow('median_sp_noise', median_sp_noise)
    cv2.imshow('median_gauss_noise', median_gauss_noise)
    cv2.imwrite('./data/median_sp_noise.jpg', median_sp_noise)
    cv2.imwrite('./data/median_gauss_noise.jpg', median_gauss_noise)
    # 高斯滤波效果
    cv2.imshow('gaussian_sp_noise', gaussian_sp_noise)
    cv2.imshow('gaussian_gauss_noise', gaussian_gauss_noise)
    cv2.imwrite('./data/gaussian_sp_noise.jpg', gaussian_sp_noise)
    cv2.imwrite('./data/gaussian_gauss_noise.jpg', gaussian_gauss_noise)
    # 均值滤波效果
    # cv2.imshow('avg_sp_noise', avg_sp_noise)
    # cv2.imshow('avg_gauss_noise', avg_gauss_noise)
    cv2.imwrite('./data/avg_sp_noise.jpg', avg_sp_noise)
    cv2.imwrite('./data/avg_gauss_noise.jpg', avg_gauss_noise)
    # 方框滤波效果
    # cv2.imshow('boxfilter_sp_noise', boxfilter_sp_noise)
    # cv2.imshow('boxfilter_gauss_noise', boxfilter_gauss_noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()