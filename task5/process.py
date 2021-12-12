# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imageProcess.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import random
import sys
import cv2 as cv
import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(QWidget):
    file_name = ''
    image = []
    opencv_img = []
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(966, 696)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 966, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_7 = QtWidgets.QMenu(self.menu_4)
        self.menu_7.setObjectName("menu_7")
        self.menu_8 = QtWidgets.QMenu(self.menu_4)
        self.menu_8.setObjectName("menu_8")
        self.menu_9 = QtWidgets.QMenu(self.menu_4)
        self.menu_9.setObjectName("menu_9")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        self.menu_6.setObjectName("menu_6")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        self.action_5 = QtWidgets.QAction(MainWindow)
        self.action_5.setObjectName("action_5")
        self.action_6 = QtWidgets.QAction(MainWindow)
        self.action_6.setObjectName("action_6")
        self.action_7 = QtWidgets.QAction(MainWindow)
        self.action_7.setObjectName("action_7")
        self.action_8 = QtWidgets.QAction(MainWindow)
        self.action_8.setObjectName("action_8")
        self.action_9 = QtWidgets.QAction(MainWindow)
        self.action_9.setObjectName("action_9")
        self.action_10 = QtWidgets.QAction(MainWindow)
        self.action_10.setObjectName("action_10")
        self.action_11 = QtWidgets.QAction(MainWindow)
        self.action_11.setObjectName("action_11")
        self.action_12 = QtWidgets.QAction(MainWindow)
        self.action_12.setObjectName("action_12")
        self.action_13 = QtWidgets.QAction(MainWindow)
        self.action_13.setObjectName("action_13")
        self.action_17 = QtWidgets.QAction(MainWindow)
        self.action_17.setObjectName("action_17")
        self.action_18 = QtWidgets.QAction(MainWindow)
        self.action_18.setObjectName("action_18")
        self.action_19 = QtWidgets.QAction(MainWindow)
        self.action_19.setObjectName("action_19")
        self.action_20 = QtWidgets.QAction(MainWindow)
        self.action_20.setObjectName("action_20")
        self.action_21 = QtWidgets.QAction(MainWindow)
        self.action_21.setObjectName("action_21")
        self.action_22 = QtWidgets.QAction(MainWindow)
        self.action_22.setObjectName("action_22")
        self.action_90 = QtWidgets.QAction(MainWindow)
        self.action_90.setObjectName("action_90")
        self.action_91 = QtWidgets.QAction(MainWindow)
        self.action_91.setObjectName("action_91")
        self.actionup_10 = QtWidgets.QAction(MainWindow)
        self.actionup_10.setObjectName("actionup_10")
        self.actionlow_10 = QtWidgets.QAction(MainWindow)
        self.actionlow_10.setObjectName("actionlow_10")
        self.actionup_11 = QtWidgets.QAction(MainWindow)
        self.actionup_11.setObjectName("actionup_11")
        self.actionlow_11 = QtWidgets.QAction(MainWindow)
        self.actionlow_11.setObjectName("actionlow_11")
        self.menu.addAction(self.action_2)
        self.menu.addAction(self.action_3)
        self.menu.addAction(self.action_4)
        self.menu_2.addAction(self.action_5)
        self.menu_2.addAction(self.action_6)
        self.menu_3.addAction(self.action_7)
        self.menu_3.addAction(self.action_8)
        self.menu_3.addAction(self.action_9)
        self.menu_3.addAction(self.action_10)
        self.menu_3.addAction(self.action_11)
        self.menu_3.addAction(self.action_12)
        self.menu_3.addAction(self.action_13)
        self.menu_7.addAction(self.action_90)
        self.menu_7.addAction(self.action_91)
        self.menu_8.addAction(self.actionup_10)
        self.menu_8.addAction(self.actionlow_10)
        self.menu_9.addAction(self.actionup_11)
        self.menu_9.addAction(self.actionlow_11)
        self.menu_4.addAction(self.menu_8.menuAction())
        self.menu_4.addAction(self.menu_7.menuAction())
        self.menu_4.addAction(self.menu_9.menuAction())
        self.menu_4.addAction(self.action_17)
        self.menu_5.addAction(self.action_18)
        self.menu_5.addAction(self.action_19)
        self.menu_5.addAction(self.action_20)
        self.menu_6.addAction(self.action_21)
        self.menu_6.addAction(self.action_22)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Processing ——By cccht"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "填充与线条"))
        self.menu_3.setTitle(_translate("MainWindow", "效果"))
        self.menu_4.setTitle(_translate("MainWindow", "大小与属性"))
        self.menu_7.setTitle(_translate("MainWindow", "位置"))
        self.menu_8.setTitle(_translate("MainWindow", "亮度"))
        self.menu_9.setTitle(_translate("MainWindow", "对比度"))
        self.menu_5.setTitle(_translate("MainWindow", "图片"))
        self.menu_6.setTitle(_translate("MainWindow", "帮助"))
        self.action_2.setText(_translate("MainWindow", "载入"))
        self.action_3.setText(_translate("MainWindow", "另存为"))
        self.action_4.setText(_translate("MainWindow", "退出"))
        self.action_5.setText(_translate("MainWindow", "填充"))
        self.action_6.setText(_translate("MainWindow", "线条"))
        self.action_7.setText(_translate("MainWindow", "模糊"))
        self.action_8.setText(_translate("MainWindow", "锐化"))
        self.action_9.setText(_translate("MainWindow", "高斯噪声"))
        self.action_10.setText(_translate("MainWindow", "椒盐噪声"))
        self.action_11.setText(_translate("MainWindow", "黑白二值"))
        self.action_12.setText(_translate("MainWindow", "饱和度"))
        self.action_13.setText(_translate("MainWindow", "艺术效果"))
        self.action_17.setText(_translate("MainWindow", "可选文字"))
        self.action_18.setText(_translate("MainWindow", "图片更正"))
        self.action_19.setText(_translate("MainWindow", "图片颜色"))
        self.action_20.setText(_translate("MainWindow", "裁剪"))
        self.action_21.setText(_translate("MainWindow", "在线网站"))
        self.action_22.setText(_translate("MainWindow", "关于"))
        self.action_90.setText(_translate("MainWindow", "逆时针 90°"))  # 逆时针参数 0
        self.action_91.setText(_translate("MainWindow", "顺时针 90°"))  # 顺时针参数 1
        self.actionup_10.setText(_translate("MainWindow", "up 10%"))
        self.actionlow_10.setText(_translate("MainWindow", "low 10%"))
        self.actionup_11.setText(_translate("MainWindow", "up 10%"))
        self.actionlow_11.setText(_translate("MainWindow", "low 10%"))
        self.action_2.triggered.connect(lambda: self.action_2_click())
        self.action_7.triggered.connect(lambda: self.blur())
        self.action_8.triggered.connect(lambda: self.sharpen())
        self.action_9.triggered.connect(lambda: self.gasuss_noise())
        self.action_10.triggered.connect(lambda: self.sp_noise())
        self.action_11.triggered.connect(lambda: self.threshold())
        self.action_90.triggered.connect(lambda: self.RotateClock(0))
        self.action_91.triggered.connect(lambda: self.RotateClock(1))
        self.actionup_10.triggered.connect(lambda: self.update_alpha(1))
        self.actionlow_10.triggered.connect(lambda: self.update_alpha(0))
        self.action_12.triggered.connect(lambda: self.change_bhd())
        # self.action_2.clicked.connect(self.action_2_click) change_bhd

    # 更新亮度
    def update_alpha(self, method):
        if method == 1:
            self.image = np.uint8(np.clip((2 * (np.int16(self.image) - 60) + 50), 0, 255))
        else:
            self.image = np.uint8(np.clip(((np.int16(self.image) - 50) / 2 + 60), 0, 255))
        # self.image = np.uint8(np.clip((2 * (np.int16(self.image) - 60) + 50), 0, 255))
        # self.image = np.hstack((self.image, res))  # 两张图片横向合并显示
        self.dis_cv_img()

    # 顺时针旋转90度
    def RotateClock(self, direction):
        self.image = cv.transpose(self.image)
        self.image = cv.flip(self.image, direction)
        self.dis_cv_img()

    # 黑白二值
    def threshold(self):
        # 执行图片灰度处理函数
        self.change_to_gray()
        # 二值化处理，低于阈值的像素点灰度值置为0；高于阈值的值置为参数3
        ret, thresh = cv.threshold(self.image, 127, 255, cv.THRESH_BINARY)
        self.image = thresh
        self.dis_cv_img()

    # 椒盐噪声
    def sp_noise(self, prob=0.05):
        '''
        添加椒盐噪声
        prob:噪声比例
        '''
        output = np.zeros(self.image.shape, np.uint8)
        thres = 1 - prob
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        self.image = output
        self.dis_cv_img()

    # 饱和度
    def change_bhd(self):
        # 加载图片 读取彩色图像归一化且转换为浮点型
        image = cv.imread(self.file_name, cv.IMREAD_COLOR).astype(np.float32) / 255.0
        print(len(image[0]))
        # 颜色空间转换 BGR转为HLS
        hlsImg = cv.cvtColor(image, cv.COLOR_BGR2HLS)
        MAX_VALUE = 10
        MAX_VALUE2 = 100

        # 调整饱和度和亮度
        # 复制原图
        hlsCopy = np.copy(hlsImg)
        # 得到 lightness 和 saturation 的值
        lightness = 1
        saturation = 1
        # 调整亮度
        hlsCopy[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
        # 饱和度
        hlsCopy[:, :, 2] = (1.0 + saturation / float(MAX_VALUE2)) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = (cv.cvtColor(hlsCopy, cv.COLOR_HLS2BGR) * 255).astype(int)

        # b, g, r = cv.split(lsImg)
        # self.image = cv.merge([r, g, b])

        cv.imwrite(self.file_name+'_temp.png', lsImg)
        self.opencv_img = cv.imread(self.file_name + '_temp.png')
        # print(opencv_img)
        # 设置图片在label展示
        if self.opencv_img is None:
            print('None')
        else:
            self.image = cv.cvtColor(self.opencv_img, cv.COLOR_BGR2RGB)
            print(self.image.data)
            self.dis_cv_img()
        # self.dis_cv_img()

    # 锐化
    def sharpen(self):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        self.image = cv.filter2D(self.image, -1, kernel=kernel)
        self.dis_cv_img()

    # 模糊
    def blur(self):
        self.image = cv.blur(self.image, (15, 1))
        self.dis_cv_img()

    def gasuss_noise(self, mean=0, var=0.001):
        """
            添加高斯噪声
            mean : 均值
            var : 方差
        """
        self.image = np.array(self.image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, self.image.shape)
        out = self.image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        self.image = out
        self.dis_cv_img()

    # 载入
    def action_2_click(self):
        # img_name = "F:/1.jpg"
        img_name, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")

        if img_name == "":
            return 0
        self.file_name = img_name
        self.label.setPixmap(QPixmap(img_name))
        self.label.setScaledContents(True)
        self.opencv_img = cv.imread(self.file_name)
        # print(opencv_img)
        # 设置图片在label展示
        if self.opencv_img is None:
            print('None')
        else:
            print("读取成功")
            self.image = cv.cvtColor(self.opencv_img, cv.COLOR_BGR2RGB)
            print(self.image.data)
            self.dis_cv_img()
            self.change_bhd()

    # 图片处理后展示
    def dis_cv_img(self):
        height, width, byteValue = self.image.shape
        print(height, width, byteValue)
        byteValue = byteValue * width

        qimage = QtGui.QImage(self.image, width, height, byteValue, QtGui.QImage.Format_RGB888)
        # self.label_2.setPixmap(QPixmap(""))
        # qt_img = QtGui.QImage(self.image.data,
        #                       self.image.shape[1],
        #                       self.image.shape[0],
        #                       self.image.shape[1] * 3,
        #                       QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap(qimage))
        self.label_2.show()
        self.label_2.setScaledContents(True)

    # 图片灰度处理
    def change_to_gray(self):
        # 灰度读取图片
        img_gray = self.opencv_img[:, :, 0] * 0.11 + self.opencv_img[:, :, 1] * 0.59 + self.opencv_img[:, :, 2] * 0.3
        self.image = cv.cvtColor(img_gray.astype(np.uint8), cv.COLOR_BGR2RGB)

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())