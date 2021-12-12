import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# OpenCV读取是按照BGR通道 PIL是按照RGB通道读取
def PILImageToCV(imagePath):

    # PIL Image转换成OpenCV格式

    img = Image.open(imagePath)

    plt.imshow(img)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    plt.imshow(img)

    plt.show()



def CVImageToPIL(imagePath):

    # OpenCV图片转换为PIL image

    img = cv2.imread(imagePath)

    plt.imshow(img)

    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.imshow(img2)

    plt.show()