import numpy as np
import cv2
import numpy as np
import time
class BOW(object):

    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.SIFT_create()
        # 是否训练标志
        self.is_train = False
        # 读取本地模型
        try:
            self.svm = cv2.ml.SVM_load("./model/svm_data.dat")
            print(self.svm)
        except Exception as e:
            print("Error: 没有找到文件或读取文件失败", e)
            self.is_train = True
    def path(self, cls, i):
        '''
        用于获取图片的全路径
        '''
        if cls == 'cat':
            return '%s/cats/%s.%d.jpg' % (self.train_path, cls, i)
        elif cls == 'dog':
            return '%s/dogs/%s.%d.jpg' % (self.train_path, cls, i)
        elif cls == 'other':
            return '%s/others/%s.%d.jpg' % (self.train_path, cls, i)
        else:
            return '%s/%s.%d.jpg' % (self.train_path, cls, i)

    def fit(self, train_path, k):
        '''
        开始训练
        args：
            train_path：训练集图片路径  我使用的数据格式为 train_path/cat/cat.i.jpg  train_path/other/other.i.jpg
            k：k-means参数k
        '''
        self.train_path = train_path
        # print(self.train_path)
        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
        # print(self.train_path)
        positive = 'dog'
        negative = 'cat'

        # 指定用于提取词汇字典的样本数
        length = 50
        # 合并特征数据  每个类从数据集中读取length张图片(length个猫)，通过聚类创建视觉词汇
        for i in range(0, length):
            # print(self.path(positive, i))
            # print(self.path(positive, i))
            # print(self.path(negative, i))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(positive, i)))
            bow_kmeans_trainer.add(self.sift_descriptor_extractor(self.path(negative, i)))
        # 进行k-means聚类，返回词汇字典 也就是聚类中心
        voc = bow_kmeans_trainer.cluster()

        # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)
        print(type(voc), voc.shape)

        # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        self.bow_img_descriptor_extractor.setVocabulary(voc)
        # 创建一个SVM对象
        if bow.is_train:
            # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
            # 按照下面的方法生成相应的正负样本图片的标签 1：正匹配  -1：负匹配
            traindata, trainlabels = [], []
            for i in range(1, length):  # 这里取400张图像做训练
                traindata.extend(self.bow_descriptor_extractor(self.path(positive, i)))
                trainlabels.append(1)
                # print(self.path(negative,i))
                traindata.extend(self.bow_descriptor_extractor(self.path(negative, i)))
                trainlabels.append(-1)
            self.svm = cv2.ml.SVM_create()
            # 设置SVM核 线性分类器
            self.svm.setType(cv2.ml.SVM_C_SVC)
            self.svm.setKernel(cv2.ml.SVM_LINEAR)
            self.svm.setC(0.01)
            # 使用训练数据和标签进行训练
            self.svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
            self.svm.save("./model/svm_data.dat")

    def predict(self, img_path):
        '''
        进行预测样本
        '''
        # 提取图片的BOW特征描述
        data = self.bow_descriptor_extractor(img_path)
        res = self.svm.predict(data)
        print(img_path, '\t', res[1][0][0])

        # 如果是猫 返回True
        if res[1][0][0] == 1.0:
            return True
        # 如果不是猫，返回False
        else:
            return False

    def sift_descriptor_extractor(self, img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        '''
        im = cv2.imread(img_path, 0)
        return self.descriptor_extractor.compute(im, self.feature_detector.detect(im))[1]

    def bow_descriptor_extractor(self, img_path):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        '''
        im = cv2.imread(img_path, 0)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


if __name__ == '__main__':
    # 测试样本数量，测试结果
    test_samples = 100
    test_results = np.zeros(test_samples, dtype=np.bool)

    # 训练集图片路径  猫和其它  进行训练
    train_path = './trains/data'
    bow = BOW()

    bow.fit(train_path, 20)
    # 指定测试图像路径
    begin_index = 4001
    for index in range(begin_index, begin_index + test_samples):
        cat = './tests/data/dogs/dog.{0}.jpg'.format(index)
        # print(cat)
        # cat_img = cv2.imread(cat)
        # 预测
        cat_predict = bow.predict(cat)
        test_results[index - begin_index] = cat_predict
    # 计算准确率
    accuracy = np.mean(test_results.astype(dtype=np.float32))
    print('测试准确率为：', accuracy * 100, '%')

    # 可视化最后一个
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(test_results)

    for index in range(begin_index, begin_index + 3):
        if test_results[index - begin_index]:
            # dog_img = cv2.imread('./tests/data/dogs/dog.{0}.jpg'.format(index))
            dog_img = cv2.imread('./tests/data/dogs/dog.{0}.jpg'.format(index))
            cv2.putText(dog_img, 'Dog Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('dog_img_{0}'.format(index), dog_img)
        else:
            cat_img = cv2.imread('./tests/data/cats/cat.{0}.jpg'.format(index))
            cv2.putText(cat_img, 'Cat Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('cat_img_{0}'.format(index), cat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
