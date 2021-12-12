import numpy as np
import random
import cv2

from matplotlib import pyplot as plt
from gray import image_gray

def sp_noise(image,prob):

    '''
    添加椒盐噪声
    prob:噪声比例
    '''

    output = np.zeros(image.shape,np.uint8)

    thres = 1 - prob

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()

            if rdn < prob:

                output[i][j] = 0

            elif rdn > thres:

                output[i][j] = 255

            else:

                output[i][j] = image[i][j]

    return output

def gauss_noise(image, mean=0, var=0.001):

    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''

    image = np.array(image/255, dtype=float)

    noise = np.random.normal(mean, var ** 0.5, image.shape)

    print("noise",noise)
    out = image + noise

    if out.min() < 0:

        low_clip = -1.

    else:

        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)

    out = np.uint8(out*255)

    #cv.imshow("gauss", out)

    return out


if __name__ == "__main__":
    # Read image
    img = cv2.cvtColor(np.asarray(image_gray("./data/test.jpeg")), cv2.COLOR_RGB2BGR)
    # img = cv2.imread("./data/test.jpg")
    # 添加椒盐噪声，噪声比例为 0.02
    sp_noise_img = sp_noise(img, prob=0.02)
    # 添加高斯噪声，均值为0，方差为0.001
    gauss_noise_img = gauss_noise(img, mean=0, var=0.001)
    # 显示原始图像
    cv2.imshow('img', img)
    cv2.imshow('add_sp_noise', sp_noise_img)
    cv2.imshow('add_gauss_noise', gauss_noise_img)
    cv2.imwrite('./data/sp_noise_img.jpg', sp_noise_img)
    cv2.imwrite('./data/gauss_noise_img.jpg', gauss_noise_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img = cv2.imread('./data/test.jpg')  # 读取通道顺序为B、G、R
    # # img[:,:,0]表示图片的蓝色通道，对一个字符串s进行翻转用的是s[::-1]，同样img[:,:,::-1]就表示BGR通道翻转，变成RGB
    # img_new2 = img[:, :, ::-1]
    #
    # plt.xticks([]), plt.yticks([])  # 隐藏x和y轴
    # plt.figure("Image out1")  # 图像窗口名称
    # plt.imshow(img_new2)
    #
    # img = cv2.imread('./data/test.jpg')
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    #
    # # plt.xticks([]), plt.yticks([])  # 隐藏x和y轴
    # plt.figure("Image out2")  # 图像窗口名称
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    # plt.imshow(img)
    # 显示图像
    # plt.figure("Image out1")  # 图像窗口名称
    # plt.imshow(out1)
    # plt.axis('on')  # 关掉坐标轴为 off
    # plt.title('image1')  # 图像题目
    # plt.figure("Image out2")  # 图像窗口名称
    # plt.imshow(out2)
    # plt.axis('on')  # 关掉坐标轴为 off
    # plt.title('image2')  # 图像题目
    # plt.show()