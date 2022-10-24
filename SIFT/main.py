from PyQt5 import QtWidgets
import sys
import cv2
import numpy as np

class Global:
    image1 = None
    image2 = None

g = Global()

class SIFT:
    def __init__(self):
        pass

    def keyPoints(self):
        if g.image1 is None:
            print("image1 is None, please input image 1 first.")
        img = cv2.imdecode(np.fromfile(g.image1, dtype=np.uint8), 1)  # type: ignore
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        keypoints, _ = sift.detectAndCompute(img, None)
        
        res = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0))
        cv2.namedWindow("4-1 keypoints", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("4-1 keypoints", res)

    def matchedKeyPoints(self):
        if g.image1 is None:
            print("image1 is None, please input image 2 first.")
        if g.image2 is None:
            print("image2 is None, please input image 2 first.")
        img1 = cv2.imdecode(np.fromfile(g.image1, dtype=np.uint8), 1)  # type: ignore
        img2 = cv2.imdecode(np.fromfile(g.image2, dtype=np.uint8), 1)  # type: ignore
        sift = cv2.SIFT_create()
        key1, des1 = sift.detectAndCompute(img1, None)
        key2, des2 = sift.detectAndCompute(img2, None)
        # reference
        # https://www.programcreek.com/python/example/110686/cv2.DescriptorMatcher_create
        # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
        matcher = cv2.DescriptorMatcher_create("FlannBased")
        rowMatched = matcher.knnMatch(des1, des2, k = 2)
        matchedOneToTwo = []
        for m in rowMatched:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.65:
                matchedOneToTwo.append(m)
        img3 = cv2.drawMatchesKnn(
            img1, key1, img2, key2, matchedOneToTwo, None, (0, 255, 255), (0, 255, 0), None) 
        cv2.namedWindow("4-2 matched keypoints", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("4-2 matched keypoints", img3)

class Window:
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html

    def __init__(self):
        self.windowHeight = 320
        self.UnitWIDTH = 250
        self.UnitWIDTHWithSpace = self.UnitWIDTH+10
        self.sift = SIFT()

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('2022 CvDl Hw1-4')
        self.window.resize(self.UnitWIDTHWithSpace*1, self.windowHeight)
        self.boxA()

    def openImage1(self):
        # https://shengyu7697.github.io/python-pyqt-qfiledialog/
        filename, _ = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.image1 = filename  # type: ignore

    def openImage2(self):
        # https://shengyu7697.github.io/python-pyqt-qfiledialog/
        filename, _ = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.image2 = filename  # type: ignore

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self.window)
        btn1.setText('Load Image 1')
        btn1.clicked.connect(self.openImage1)

        btn2 = QtWidgets.QPushButton(self.window)
        btn2.setText('Load Image 2')
        btn2.clicked.connect(self.openImage2)

        btn3 = QtWidgets.QPushButton(self.window)
        btn3.setText('4.1 Keypoints')
        btn3.clicked.connect(self.sift.keyPoints)

        btn4 = QtWidgets.QPushButton(self.window)
        btn4.setText('4.2 Matched Keypoints')
        btn4.clicked.connect(self.sift.matchedKeyPoints)

        box = QtWidgets.QGroupBox(title="4 SIFT", parent=self.window)
        box.setGeometry(0, 0, self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addWidget(btn4)

    def render(self):
        self.window.show()
        sys.exit(self.app.exec())

window = Window()
window.render()
