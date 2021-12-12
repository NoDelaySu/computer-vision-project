import cv2
import numpy as np
from matplotlib import pyplot as plt
#计算Kirsch 边沿检测算子
def Kirsch(img):

    #定义Kirsch 卷积模板
    m1 = np.array([[5, 5, 5],[-3,0,-3],[-3,-3,-3]])

    m2 = np.array([[-3, 5,5],[-3,0,5],[-3,-3,-3]])

    m3 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])

    m4 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])

    m5 = np.array([[-3, -3, -3],[-3,0,-3],[5,5,5]])

    m6 = np.array([[-3, -3, -3],[5,0,-3],[5,5,-3]])

    m7 = np.array([[5, -3, -3],[5,0,-3],[5,-3,-3]])

    m8 = np.array([[5, 5, -3],[5,0,-3],[-3,-3,-3]])

    # img = cv2.imread("./data/test.jpg",0)

    # img = cv2.GaussianBlur(img,(3,3),5)

    #周围填充一圈
    #卷积时，必须在原图周围填充一个像素
    img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)

    temp = list(range(8))

    img1 = np.zeros(img.shape) #复制空间  此处必须的重新复制一块和原图像矩阵一样大小的矩阵，以保存计算后的结果

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            temp[0] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )
                #利用矩阵的二次型表达，可以计算出矩阵的各个元素之和
            temp[1] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            temp[2] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )

            temp[3] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m3 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            temp[4] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m4 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            temp[5] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m5 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            temp[6] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m6 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            temp[7] = np.abs(
                (np.dot(np.array([1, 1, 1]), (m7 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )

            img1[i,j] = np.max(temp)

            if img1[i, j] > 255:  #此处的阈值一般写255，根据实际情况选择0~255之间的值
                img1[i, j] = 255
            else:
                img1[i, j] = 0
    return img1

if __name__ == '__main__':
    img = cv2.imread("./data/test.jpg", 0)
    img1 = Kirsch(img)
    cv2.imshow("result",img1)
    print(img.shape)
    cv2.waitKey(0)

