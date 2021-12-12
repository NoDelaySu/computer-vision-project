import cv2
import numpy as np
from matplotlib import pyplot as plt


def sys_equalizehist(img):
    '''
    利用系统自带的函数进行直方图均衡化
    :param img: 待处理图像
    :return:  [equ_img,equ_hist]，返回一个列表，第一个元素是均衡化后的图像，第二个是均衡了的直方图
    '''
    img = cv2.imread(img, 0)
    h, w = img.shape
    equ_img = cv2.equalizeHist(img)  # 得到直方图均衡化后的图像
    equ_hist = cv2.calcHist([equ_img], [0], None, [256], [0, 255])  # 得到均衡化后的图像的灰度直方图
    equ_hist[0:255] = equ_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
    # res = np.hstack((img,equ)) #stacking images side-by-side#这一行是将两个图像进行了行方向的叠加
    return [equ_img, equ_hist]

def def_equalizehist(img, L=256):
    '''
    根据均衡化原理自定义函数
    :param img: 待处理图像
    :param L: 灰度级别的个数
    :return: [equal_img,equal_hist]返回一个列表，第一个元素是均衡化后的图像，第二个是均衡了的直方图
    '''
    img = cv2.imread(img, 0)

    # 第一步获取图像的直方图
    h, w = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])  # 这里返回的是次数
    hist[0:255] = hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式

    # 第二步得到灰度级概率累积直方图
    sum_hist = np.zeros(hist.shape)  # 用于存放灰度级别概率的累和
    for i in range(256):
        sum_hist[i] = sum(hist[0:i + 1])  # 将前i+1个灰度级别的出现概率总和赋值给sum_hist[i]

    # 第三步通过映射函数获得原图像灰度级与均衡后图像的灰度级的映射关系，这里创建映射后的灰度级别排序
    equal_hist = np.zeros(sum_hist.shape)
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)

    # 第四步根据第三步的映射关系将灰度图每个像素点的灰度级别替换为映射后的灰度级别，这里是这样换的，equal_hist的索引号相当于原先的灰度级别排序，元素值则是映射后的灰度级别
    equal_img = img.copy()  # 用于存放均衡化后图像的灰度值
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]
    # 计算得到均衡化后的直方图
    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 255])
    equal_hist[0:255] = equal_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
    return [equal_img, equal_hist]


if __name__ == '__main__':
    img = "./data/test.png"
    sys_img, sys_hist = sys_equalizehist(img)
    def_img, def_hist = def_equalizehist(img)
    x = np.linspace(0, 255, 256)
    plt.subplot(1, 2, 1), plt.plot(x, sys_hist, '-b')
    plt.subplot(1, 2, 2), plt.plot(x, def_hist, '-r')
    plt.show()