import numpy  as np
from PIL import Image
import matplotlib.pyplot as plt

"""定义Sobel算子的算法"""

def SobelOperator(roi, operator_type):
    # 判断模板为横向或纵向
    if operator_type == "horizontal":
        sobel_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == "vertical":
        sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        raise ("type Error")
    # 像素与对应模板相乘求和的绝对值
    result = np.abs(np.sum(roi * sobel_operator))
    return result


def sobel(inp, oup):
    # 读入图片并转化为灰度图
    img_gray = Image.open(inp).convert('L')
    # 将灰度图转化为数组，读入长和宽
    img_array = np.array(img_gray)
    height, width = img_array.shape
    # 创建一个新的数组用来存储计算新产生的像素值
    img_edge = np.zeros((height - 1, width - 1))
    # 为新数组填入新的像素值
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            Sx = SobelOperator(img_array[x - 1:x + 2, y - 1:y + 2], "horizontal")
            Sy = SobelOperator(img_array[x - 1:x + 2, y - 1:y + 2], "vertical")
            img_edge[x][y] = (Sx * Sx + Sy * Sy) ** 0.5
    img_edge_new = Image.fromarray(img_edge).convert('L')
    plt.subplot(121)
    plt.title('原图', fontproperties='SimHei')
    plt.imshow(img_gray, cmap='gray')
    plt.subplot(122)
    plt.title('边缘检测', fontproperties='SimHei')
    plt.imshow(img_edge_new, cmap='gray')
    img_edge_new.save(oup)
    plt.show()



if __name__ ==  '__main__':
    # main函数
    inp = './data/test.jpg'
    oup = './data/sobel_detect2.jpg'
    sobel(inp, oup)
