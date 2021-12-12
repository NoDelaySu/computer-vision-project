# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def General_line(x1, y1, x2, y2):
    # 一般式 Ax+By+C=0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    k = -1 * A / B
    b = -1 * C / B
    return k, b

#读取图像
img = cv2.imread('./data/test.jpeg')

#灰度转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#转换为二值图像
edges = cv2.Canny(gray, 50, 200)

#显示原始图像
plt.subplot(121), plt.imshow(edges, 'gray'), plt.title(u'Canny边缘检测图像')
plt.axis('off')

#霍夫变换检测直线
minLineLength = 60
maxLineGap = 10
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength, maxLineGap)
# lines = cv2.HoughLinesP(edges, 1, (1 * np.pi / 180), 18, minLineLength=150, maxLineGap=10)
lines = cv2.HoughLinesP(edges, 1, (1 * np.pi / 180), 18, minLineLength=200, maxLineGap=5)

#绘制直线
lines1 = lines[:, 0, :]
for x1, y1, x2, y2 in lines1[:3]: # 选择最近3条直线
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    k, b = General_line(x1, y1, x2, y2)  # 求直线方程k,b
    print("y="+str(k)+"x+"+str(b))  # 控制台打印直线方程

res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#显示处理图像
plt.subplot(122), plt.imshow(res), plt.title(u'Hough检测结果图像')
plt.axis('off')
plt.show()