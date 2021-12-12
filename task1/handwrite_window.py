import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt


class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.resize(284, 330)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.pos_xy = []  #保存鼠标移动过的点

        # 添加一系列控件
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('识别结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("关闭", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def imageprepare(self, image):
        plt.imshow(image)  # 显示需要识别的图片
        plt.show()
        image = image.convert('L')
        tv = list(image.getdata())
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        return tva

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
        '''
        #中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        #pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 380, 380)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素

        recognize_result = self.recognize_img(im)  # 调用识别函数

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):  # 手写体识别函数
        return self.predict(img)
        # myimage = img.convert('L')  # 转换成灰度图
        # tv = list(myimage.getdata())  # 获取图片像素值
        # tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑
        #
        # init = tf.global_variables_initializer()
        # saver = tf.train.Saver
        # x = tf.placeholder(tf.float32, [None, 784])
        #
        # y_ = tf.placeholder(tf.float32, [None, 10])
        # with tf.Session() as sess:
        #     sess.run(init)
        #     saver = tf.train.import_meta_graph('./model/model.ckpt.meta')  # 载入模型结构
        #     saver.restore(sess, './model/model.ckpt')  # 载入模型参数
        #
        #     graph = tf.get_default_graph()  # 加载计算图
        #     x = graph.get_tensor_by_name("x:0")  # 从模型中读取占位符变量
        #     keep_prob = graph.get_tensor_by_name("keep_prob:0")
        #     y_conv = graph.get_tensor_by_name("y_conv:0")  # 关键的一句  从模型中读取占位符变量
        #
        #     prediction = tf.argmax(y_conv, 1)
        #     predint = prediction.eval(feed_dict={x: [tva], keep_prob: 1.0}, session=sess)  # feed_dict输入数据给placeholder占位符
        #     print(predint[0])
        # return predint[0]

    def predict(self, image):

        result = self.imageprepare(image)
        x = tf.placeholder(tf.float32, [None, 784])

        y_ = tf.placeholder(tf.float32, [None, 10])
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()

        predict_result = None
        # *************** 开始预测 *************** #
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "./model/model.ckpt")  # 使用模型，参数和之前的代码保持一致
            prediction = tf.argmax(y_conv, 1)
            predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)
            print('识别结果:')
            print(predint[0])
            predict_result = predint[0]
        tf.reset_default_graph()
        return predict_result