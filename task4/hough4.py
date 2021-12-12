import cv2
import numpy as np
#hough变换的程序实现

img = cv2.imread("./data/test.png")#读取图片
img2 = img.copy()
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#将图片转换为灰度图
edges = cv2.Canny(gray,50,225,apertureSize = 3)#canny算法提取轮廓

#基于概率的hough变换......................................................
lines_Probabilitys = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=50,maxLineGap=5)#概率hough变换
lines_Probability = lines_Probabilitys[:,0,:]#提取为二维
for x1,y1,x2,y2 in lines_Probability[:]:
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)

cv2.namedWindow("HoughLines_Probabilitys", 2)   #创建一个窗口
cv2.imshow('HoughLines_Probabilitys', img)    #显示原始图片

#标准的hough变换......................................................
lines_standards = cv2.HoughLines(edges,1,np.pi/180,200) #标准hough变换查找直线

#绘制hough变换后找到的所有直线，返回数据是一个二位数组
for lines_standard in lines_standards:
    for rho,theta in lines_standard:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)

print(lines_standards)#打印出找到的直线的极坐标系坐标、

cv2.namedWindow("HoughLines_standards", 2)   #创建一个窗口
cv2.imshow('HoughLines_standards', img2)    #显示原始图片

cv2.waitKey()
