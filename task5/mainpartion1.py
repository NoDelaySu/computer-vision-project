from cv2 import *
from numpy import *
from PIL import Image

"""
这里做的是一个单通道图的主成分分析，三通道的等以后用到了在写，学习进度所限
还有一个感受就是，opencv用着比PIL要复杂，
因为第一遍使用了opencv，但在最后转成图像的时候，opencv出了问题，所以我将开头和结尾改用了PIL
"""

img = Image.open("./data/suns.jpg")

# 图像转矩阵
data = img.convert("L")
data = mat(data)  # np.mat和np.array的区别在于，*可以直接表示矩阵乘法，等价于np.dot()

# 去中心化
mean_x = mean(data, axis=1)  # 按行计算均值
data = data.astype("float64")  # 就是说float64和unit8不能直接相减，所以要转换下
data -= mean_x

# 协方差矩阵计算
cov_mat = 1.0 / (data.shape[0] - 1) * data * data.T

# 求协方差矩阵的特征值特征向量
evalue, evector = linalg.eig(cov_mat)

# 寻找主要特征值对应的特征向量，因为协方差矩阵和对角矩阵特征值相同，所以对应的特征向量的位置一样，所以可通过对角阵的主要特征值找出特征向量
evalue_index = argsort(evalue)[::-1]  # argsor是按从小到大的顺序排列，所以取反

k = input("你想要的保留的维度：")
evalue_index = evalue_index[:int(k)]
principal_evector = evector[:, evalue_index]

# 信息损失率


# 获取原始数据在改组基上的坐标
loc = principal_evector.T.dot(data)
# 获取原始数据在本组基下的表现形式
recover_mat = principal_evector.dot(loc)

# 将矩阵数据转成图像
recove_image = recover_mat + mean_x
newImg = Image.fromarray(uint8(recove_image))
newImg.show()
