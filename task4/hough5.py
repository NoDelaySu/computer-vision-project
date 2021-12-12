import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):  # 滑动条的回调函数
    pass


src = cv2.imread('./data/test.png')
srcBlur = cv2.GaussianBlur(src, (3, 3), 0)
gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
WindowName = 'Approx'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口

cv2.createTrackbar('threshold', WindowName, 0, 100, nothing)  # 创建滑动条
cv2.createTrackbar('minLineLength', WindowName, 0, 100, nothing)  # 创建滑动条
cv2.createTrackbar('maxLineGap', WindowName, 0, 100, nothing)  # 创建滑动条

while (1):
    img = src.copy()
    threshold = cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值
    minLineLength = 2 * cv2.getTrackbarPos('minLineLength', WindowName)  # 获取滑动条值
    maxLineGap = cv2.getTrackbarPos('maxLineGap', WindowName)  # 获取滑动条值

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow(WindowName, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()