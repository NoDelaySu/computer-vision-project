import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def image_gray(inp):
    imarry = np.asarray(Image.open(inp))  # 将图片转换成numpy数组
    height, width, channel = imarry.shape
    print(channel)
    imarry_gray = np.zeros((height, width))  # 初始化imarry_gray
    for i in range(height):
        for j in range(width):
            imarry_gray[i][j] = int(0.3 * imarry[i][j][0] + 0.59 * imarry[i][j][1] + 0.11 * imarry[i][j][2])
    # image_gray1 = Image.fromarray(imarry_gray)
    print("imarry_gray",imarry_gray[0].size)
    image_gray1 = Image.fromarray(imarry_gray)  # 转换成图片
    plt.imshow(image_gray1)
    plt.show()
    image_gray1 = image_gray1.convert('L')
    image_gray1.save(img_out)
    print("image_gray1", image_gray1)
    return image_gray1

img_path = './data/test.jpg'
img_out = './data/img_gray.jpg'
gray = image_gray(img_path)
gray = cv2.cvtColor(np.asarray(gray), cv2.COLOR_RGB2BGR)
cv2.imshow('gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
