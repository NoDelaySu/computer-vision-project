import numpy as np
import matplotlib.pyplot as plt
import cv2


def getLine(x0, y0, angs_resolution=100):
    """
    x0: 原空间下点横坐标
    y0: 原空间下点纵坐标
    angs_resolution: \theta 划分精度
    功能：原空间(x0,y0)点，变换到 d, \theta 参数空间的曲线。
    说明：因为计算机存储是离散值，所以只是 \theta 取到一些值下的直线。当然，\theta 取值越多，越精细。
    """

    angs = np.linspace(0, 2 * np.pi, angs_resolution)  # 定义\theta 取到的离散值
    d = x0 * np.cos(angs) + y0 * np.sin(angs)
    return angs, d


def Hough(edgeImg, angsDiv=500, dDiv=1000):
    # 获取图像尺寸
    ySize, xSize = edgeImg.shape

    # 得到二值图中所有点坐标(x,y)
    y, x = np.where(edgeImg != 0)

    # 大致确定 d 范围
    dMax = np.sqrt(np.max(y) ** 2 + np.max(x) ** 2)

    # 分辨率
    d_res = 2 * dMax / (dDiv - 1)
    ang_res = 2 * np.pi / (angsDiv - 1)

    # 亮度模板，起初为全黑，当经过某点，亮度 +1
    template = np.zeros((dDiv, angsDiv), dtype=np.uint8)

    # 亮度叠加计算
    for xx in range(len(x)):
        _, _d = getLine(x[xx], y[xx], angsDiv)
        _n = ((_d + dMax) / d_res).astype(np.int64) # d经过分辨率缩放
        angle = np.arange(0, angsDiv) # theta 0-360
        # print(*zip(angle, _n))
        for p in zip(angle, _n):
            template[p[1], p[0]] = template[p[1], p[0]] + 1
    return template


if __name__ == "__main__":
    # 读取图片
    imgPath = "./data/test.jpg"
    grayImg = cv2.imread(imgPath)

    # 提取边缘
    # edgeImg = cv2.Canny(grayImg, 50, 150)
    edgeImg = cv2.Canny(grayImg, 50, 150, apertureSize=3)
    # 设置离散的精度
    angsDiv = 360
    dDiv = 180
    # 霍夫变换
    forceImg = Hough(edgeImg, angsDiv, dDiv)
    plt.imshow(edgeImg, 'gray')
    plt.show()
    # plt.xlabel("θ")
    # plt.ylabel("ρ")
    # plt.title("ρ-θ space and peaks")
    plt.imshow(forceImg, 'gray')
    plt.show()

    # cv2.imshow('edgeImg', edgeImg)
    # cv2.imshow('forceImg', forceImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 一些后面要用到的常数
    y, x = np.where(edgeImg != 0)
    dMax = np.sqrt(np.max(y) ** 2 + np.max(x) ** 2)
    d_res = 2 * dMax / (dDiv - 1)
    ang_res = 2 * np.pi / (angsDiv - 1)

    # 这是只考察了最大点，即亮度最大的点
    ind = np.where(forceImg == np.max(forceImg))

    # theta 和 d 的真实值
    theta = (ind[1]) * ang_res
    d = -dMax + (ind[0]) * d_res

    # 对应原空间的直线
    xx = np.arange(512)
    i = 0
    yy = (d[i] - xx * np.cos(theta[i])) / np.sin(theta[i])

    plt.plot(xx, yy, "r", linewidth=1)
    plt.imshow(edgeImg, 'gray')
    plt.show()
