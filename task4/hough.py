import cv2
import numpy as np
from PIL import Image


# 根据已知两点坐标，求过这两点的直线解析方程： a*x+b*y+c = 0  (a >= 0)
def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


# 根据直线的起点与终点计算出平行距离D的平行线的方程
# def getLinearEquation(p1x, p1y, p2x, p2y, distance):
#     """
#     :param p1x: 起点X
#     :param p1y: 起点Y
#     :param p2x: 终点X
#     :param p2y: 终点Y
#     :param distance: 平距
#     :param left_right: 向左还是向右
#     """
#     e = getLinearEquation(p1x, p1y, p2x, p2y)
#     f = distance * math.sqrt(e.a * e.a + e.b * e.b)
#     m1 = e.c + f
#     m2 = e.c - f
#     # result = 值1 if 条件 else 值2
#     c2 = m1 if p2y - p1y < 0 else m2
#     return [e.a, e.b, c2]

def General_line(x1, y1, x2, y2):
    # 一般式 Ax+By+C=0
    A = y2 - y1
    B = x2 - x1
    C = x2 * y1 - x1 * y2
    k = A / B
    b = C / B
    return k, b

def hough2(inp, oup):
    # original_img = cv2.imread(inp, 0)
    original_img = cv2.imread(inp)
    img = cv2.resize(original_img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 89)  # 这里对最后一个参数使用了经验型的值
    lines = cv2.HoughLinesP(edges, 1, (1 * np.pi / 180), 18, minLineLength=220, maxLineGap=5)
    result = img.copy()
    print(len(lines))
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # k, b = General_line(x1, y1, x2, y2)  # 求直线方程k,b
            a, b, c = getLinearEquation(x1, y1, x2, y2)  # 求直线方程k,b
            print(str(a) + "x + " + str(b) + "y + " + str(c) + " = 0")  # 控制台打印直线方程
        pass
    # for line in lines:
    #     rho = line[0][0]  # 第一个元素是距离rho
    #     theta = line[0][1]  # 第二个元素是角度theta
    #     print(rho)
    #     print(theta)
    #     if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
    #         pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
    #         # 该直线与最后一行的焦点
    #         pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
    #         cv2.line(result, pt1, pt2, (255, 0, 0), 1)  # 绘制一条白线
    #     else:  # 水平直线
    #         pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
    #         # 该直线与最后一列的交点
    #         pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
    #         cv2.line(result, pt1, pt2, (255, 0, 0), 1)  # 绘制一条直线
    # 设置字体

    cv2.imshow('Canny', edges)
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    edges = Image.fromarray(edges).convert('L')
    # result = Image.fromarray(result).convert('L')
    # result.save(oup)
    edges.save("./data/rode_example_edges.jpg")
    cv2.imwrite("./data/result_example_edges.jpg", result)


if __name__ == '__main__':

    # main
    inp = "./data/test1.jpeg"
    oup = "./data/hough_detect.jpg"
    hough2(inp, oup)
