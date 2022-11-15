from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Global:
    image = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    label = None
    qt_img_label = None


g = Global()


class VGG19:
    def __init__(self):
        pass

    def showTrainImages(self):
        (g.x_train, g.y_train), (g.x_test,
                                 g.y_test) = tf.keras.datasets.cifar10.load_data()
        if g.y_train is None or g.x_train is None:
            print("Please load image first")
            return

        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.title(g.class_names[g.y_train[i][0]])
            plt.imshow(g.x_train[i])
        plt.show()

    def showModelStructure(self):
        model = tf.keras.applications.VGG19(
            include_top=True, weights=None, input_shape=(32, 32, 3), classes=10)
        model.summary()

    def showDataAugmentation(self):
        if g.image is None:
            print("Please load image first.")
            return
        img = cv2.imdecode(np.fromfile(g.image, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        # Add the image to a batch.
        img = tf.cast(tf.expand_dims(img, 0), tf.float32)

        data_augmentation1 = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2)
        ])
        data_augmentation2 = tf.keras.Sequential([
            tf.keras.layers.RandomZoom(0.5)
        ])
        data_augmentation3 = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
        ])

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 3, 1)
        plt.title('random rotation')
        plt.imshow(data_augmentation1(img)[0])
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title('random resized crop')
        plt.imshow(data_augmentation2(img)[0])
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title('random horizontal flip')
        plt.imshow(data_augmentation3(img)[0])
        plt.axis("off")
        plt.show()

    def inference(self):
        if g.image is None:
            print("Please load image first.")
            return
        pixmap = QtGui.QPixmap(g.image)
        pixmap = pixmap.scaled(200, 200)
        g.qt_img_label.setPixmap(pixmap)

        img = cv2.imdecode(np.fromfile(g.image, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        # Add the image to a batch.
        img = tf.cast(tf.expand_dims(img, 0), tf.float32)
        model = tf.saved_model.load('./vgg19')
        ans = model(img)
        index = np.argmax(ans[0])
        print(g.class_names[index], str(ans[0][index].numpy()*100)+'%')
        g.label.setText('Confidence = %.2f\nPrediction Label: %s' %
                        (ans[0][index].numpy(), g.class_names[index]))

    def loadImage(self):
        # https://shengyu7697.github.io/python-pyqt-qfiledialog/
        filename, _ = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.image = filename
        pixmap = QtGui.QPixmap(g.image)
        pixmap = pixmap.scaled(200, 200)
        g.qt_img_label.setPixmap(pixmap)

    def showAccAndLoss(self):
        pixmap = QtGui.QPixmap('train_acc_loss.jpg')
        pixmap = pixmap.scaled(500, 500)
        g.qt_img_label.setPixmap(pixmap)


class Window:
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html

    def __init__(self):
        self.windowHeight = 720
        self.UnitWIDTH = 250
        self.UnitWIDTHWithSpace = self.UnitWIDTH+10
        self.vgg19 = VGG19()

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('2022 CvDl Hw1-5')
        self.window.resize(self.UnitWIDTHWithSpace*3, self.windowHeight)
        self.showImg = None
        self.boxA()
        self.boxB()

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self.window)
        btn1.setText('Load Image')
        btn1.clicked.connect(self.vgg19.loadImage)

        btn2 = QtWidgets.QPushButton(self.window)
        btn2.setText('1. Show Train Images')
        btn2.clicked.connect(self.vgg19.showTrainImages)

        btn3 = QtWidgets.QPushButton(self.window)
        btn3.setText('2. Show Model Structure')
        btn3.clicked.connect(self.vgg19.showModelStructure)

        btn4 = QtWidgets.QPushButton(self.window)
        btn4.setText('3. Show Data Augmentation')
        btn4.clicked.connect(self.vgg19.showDataAugmentation)

        btn5 = QtWidgets.QPushButton(self.window)
        btn5.setText('4. Show Accuracy and Loss')
        btn5.clicked.connect(self.vgg19.showAccAndLoss)

        btn6 = QtWidgets.QPushButton(self.window)
        btn6.setText('5. Inference')
        btn6.clicked.connect(self.vgg19.inference)

        box = QtWidgets.QGroupBox(title="5. VGG19 Test", parent=self.window)
        box.setGeometry(0, 0, self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addWidget(btn4)
        layout.addWidget(btn5)
        layout.addWidget(btn6)

    def boxB(self):
        g.qt_img_label = QtWidgets.QLabel(self.window)
        g.label = QtWidgets.QLabel(self.window)
        g.label.resize(100, 50)
        hbox = QtWidgets.QWidget(self.window)
        hbox.setGeometry(self.UnitWIDTHWithSpace, 0, 500, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(hbox)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(g.label)
        layout.addWidget(g.qt_img_label)

    def render(self):
        self.window.show()
        sys.exit(self.app.exec())


window = Window()
window.render()
