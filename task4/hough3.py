from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def General_line(x1, y1, x2, y2):
    # 一般式 Ax+By+C=0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    k = -1 * A / B
    b = -1 * C / B
    return k, b


def hough(img):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 160)
    lines1 = lines[:, 0, :]  # 提取为为二维

    line_len = 1000
    for rho, theta in lines1[:3]:  # 显示三条直线
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + line_len * (-b))
        y1 = int(y0 + line_len * (a))
        x2 = int(x0 - line_len * (-b))
        y2 = int(y0 - line_len * (a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 最后一个参数是直线粗细
        k, b = General_line(x1, y1, x2, y2)  # 求直线方程k,b
        print("y="), print(k), print("x+"), print(b)  # 控制台打印直线方程
    cv2.imshow("image-lines", img)


img = cv2.imread('./data/test.jpg')
#灰度转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hough(gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
