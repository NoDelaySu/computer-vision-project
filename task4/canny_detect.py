from cv2 import cv2
import numpy as np
from PIL import Image, ImageEnhance

def img_processing(img):
    # 图片灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow("binary", binary)
    # 检测边缘
    edges = cv2.Canny(binary, ret-30, ret+30, apertureSize=3)
    return edges

def line_detect(img_path):
    img = Image.open(img_path)
    img = ImageEnhance.Contrast(img).enhance(3)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = img_processing(img)
    lines = cv2.HoughLinesP(result, 1, (1 * np.pi / 180), 10, minLineLength=10, maxLineGap=5)
    print("Line Num:", len(lines))

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        pass
    img = np.array(img)
    cv2.imshow("img", img)
    cv2.imshow("result", result)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = Image.fromarray(img, 'RGB')
    # img.show()


if __name__== "__main__":

    line_detect("./data/test.jpg")
    # img = cv2.imread("./data/test.jpg", 0)
    # cv2.imwrite("./data/line_detect1.jpg", cv2.Canny(img, 200, 300))
    # cv2.imshow("line detect", cv2.imread("./data/line_detect1.jpg"))

    cv2.waitKey()
    cv2.destroyAllWindows()