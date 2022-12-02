from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import os.path as path


class Global:
    imagePath = ''
    qt_img_label = None
    label = ''


g = Global()


# I trained my model on Kaggle
# https://www.kaggle.com/yutong0807/img-classification-with-tf-resnet50

class Util:
    def showImage():
        folder = './inference_dataset/'
        catFolder = path.join(folder, 'Cat/')
        dogFolder = path.join(folder, 'Dog/')
        catFiles = [f for f in os.listdir(
            catFolder) if path.isfile(path.join(catFolder, f))]
        dogFiles = [f for f in os.listdir(
            dogFolder) if path.isfile(path.join(dogFolder, f))]
        if len(catFiles) == 0 or len(dogFiles) == 0:
            print("Either dog or cat cannot be found")
            return
        cat = cv2.imdecode(np.fromfile(
            path.join(catFolder, catFiles[0]), dtype=np.uint8), 1)
        dog = cv2.imdecode(np.fromfile(
            path.join(dogFolder, dogFiles[0]), dtype=np.uint8), 1)

        # resize and convert from BGR to RGB
        cat = cv2.cvtColor(cv2.resize(cat, (224, 224)), cv2.COLOR_BGR2RGB)
        dog = cv2.cvtColor(cv2.resize(dog, (224, 224)), cv2.COLOR_BGR2RGB)

        # show figure
        plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(cat)
        plt.title("cat")
        plt.subplot(1, 2, 2)
        plt.imshow(dog)
        plt.title("dog")
        plt.show()

    def showDistribution():
        img = cv2.imdecode(np.fromfile(
            './distribution.png', dtype=np.uint8), 1)
        cv2.imshow('show distribution', img)

        # At home:
        # folder = './training_dataset/'
        # catFolder = path.join(folder, 'Cat/')
        # dogFolder = path.join(folder, 'Dog/')
        # catFiles = [f for f in os.listdir(
        #     catFolder) if path.isfile(path.join(catFolder, f))]
        # dogFiles = [f for f in os.listdir(
        #     dogFolder) if path.isfile(path.join(dogFolder, f))]

        # plt.bar(['cat', 'dog'], [len(catFiles), len(dogFiles)])
        # plt.title('Class Distribution')
        # plt.ylabel('Number of images')
        # plt.show()

    def showModelStructure():
        model = tf.keras.Sequential()
        resnet = tf.keras.applications.resnet50.ResNet50(
            include_top=False, input_shape=(224, 224, 3), weights='imagenet')
        model.add(resnet)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.summary(expand_nested=True)

    def showComparison():
        img = cv2.imdecode(np.fromfile(
            './accuracy_comparison.png', dtype=np.uint8), 1)
        cv2.imshow('accuracy comparison', img)

        # At home:
        # x = ['Binary Cross Entropy', 'Focal Loss']
        # y = [94.49, 91.76]
        # plt.bar(x, y)
        # plt.text(0,95, "94.49")
        # plt.text(1,92, "91.76")
        # plt.title('Accuracy Comparison')
        # plt.ylabel('Accuracy(%)')
        # plt.ylim(0, 100)
        # plt.show()

    def loadImage():
        # https://shengyu7697.github.io/python-pyqt-qfiledialog/
        filename, _ = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.imagePath = filename
        pixmap = QtGui.QPixmap(g.imagePath)
        pixmap = pixmap.scaled(224, 224)
        g.qt_img_label.setPixmap(pixmap)

    def inference():
        if g.imagePath == '':
            print("please load image first")
            return

        # show image on the window
        pixmap = QtGui.QPixmap(g.imagePath)
        pixmap = pixmap.scaled(224, 224)
        g.qt_img_label.setPixmap(pixmap)

        # load image and inference
        img = cv2.imdecode(np.fromfile(g.imagePath, dtype=np.uint8), 1)
        model = tf.saved_model.load('./resnet50-BinaryCrossEntrop')
        # convert to RGB and resize to (224, 224)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        # img -> batch
        img = np.reshape(img, (1,)+img.shape)
        img = img.astype(np.float32)
        ans = model(img)
        print(ans)
        g.label.setText('Predicted: %s' % ('Dog' if ans[0][0] > .5 else 'Cat'))


class Window(QtWidgets.QWidget):
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html
    # https://shengyu7697.github.io/python-pyqt-qfiledialog/
    # https://zhuanlan.zhihu.com/p/75561654

    def __init__(self):
        super().__init__()

        self.app = QtWidgets.QApplication(sys.argv)
        self.setWindowTitle('2022 CvDl HW2')

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.boxA())
        layout.addWidget(self.boxB())

        self.setLayout(layout)

    # def loadImage(self):
    #     filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
    #     if filename:
    #         g.imagePath = filename

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self)
        btn1.setText('Load Image')
        btn1.clicked.connect(Util.loadImage)

        btn2 = QtWidgets.QPushButton(self)
        btn2.setText('1. Show Images')
        btn2.clicked.connect(Util.showImage)

        btn3 = QtWidgets.QPushButton(self)
        btn3.setText('2. Show Distribution')
        btn3.clicked.connect(Util.showDistribution)

        btn4 = QtWidgets.QPushButton(self)
        btn4.setText('3. Show Model Structure')
        btn4.clicked.connect(Util.showModelStructure)

        btn5 = QtWidgets.QPushButton(self)
        btn5.setText('4. Show Comparison')
        btn5.clicked.connect(Util.showComparison)

        btn6 = QtWidgets.QPushButton(self)
        btn6.setText('5. Inference')
        btn6.clicked.connect(Util.inference)

        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn1)
        box.addWidget(btn2)
        box.addWidget(btn3)
        box.addWidget(btn4)
        box.addWidget(btn5)
        box.addWidget(btn6)

        group = QtWidgets.QGroupBox(title="5. ResNet50", parent=self)
        group.setLayout(box)

        return group

    def boxB(self):
        g.qt_img_label = QtWidgets.QLabel(self)
        g.label = QtWidgets.QLabel(self)

        box = QtWidgets.QVBoxLayout()
        box.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        box.addWidget(g.qt_img_label)
        box.addWidget(g.label)

        widget = QtWidgets.QWidget(self)
        widget.setLayout(box)

        return widget


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
