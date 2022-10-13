from os import listdir
from os.path import isfile, join, splitext
from PyQt5 import QtWidgets
import sys
import cv2
import numpy as np


class Global:
    fileList = []
    fileListMapping = {}
    extrinsicAttentionImageIndex = 0
    projectWordInput = 'CAMERA'

    def extrinsicAttentionImage(self, i):
        self.extrinsicAttentionImageIndex = i

    def listenProjectWordsInputField(self, text):
        self.projectWordInput = text.upper()


g = Global()

class Calibration:
    def __init__(self):
        self.boardW = 11
        self.boardH = 8
        self.result = ()

    def findCorders(self):
        for file in g.fileList:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
            width, height = img.shape[0], img.shape[1]
            retval, corners = cv2.findChessboardCorners(
                img, (self.boardW, self.boardH), None)
            if retval:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(
                    img, (self.boardW, self.boardH), corners, retval)
            # width : height = 100 : newHeight
            height = int(height * 1000 / width)
            width = 1000
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_NEAREST)
            cv2.imshow('preview', img)
            cv2.waitKey(500)

    def calibrateCamera(self):
        # reference:
        # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
        imagePoints = []  # 2d point in real world space
        objectPoints = []  # 3d points in image plane.
        for file in g.fileList:
            imagePoint = []
            objectPoint = []
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
            width, height = img.shape[0], img.shape[1]
            retval, corners = cv2.findChessboardCorners(
                img, (self.boardW, self.boardH), None)
            if retval:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                for (j, corner) in enumerate(corners):
                    imagePoint.append([corner[0][0], corner[0][1]])
                    objectPoint.append(
                        [int(j/self.boardW), j % self.boardW, 0])
            imagePoints.append(imagePoint)
            objectPoints.append(objectPoint)

        # retval, cameraMatrix, distCoeffs, rvecs, tvecs
        self.result = cv2.calibrateCamera(
            np.array(objectPoints, dtype='float32'),
            np.array(imagePoints, dtype='float32'),
            (width, height), None, None)
        return self.result

    def findIntrinsicMatrix(self):
        if not self.result:
            self.calibrateCamera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.result
        print(cameraMatrix)
        return cameraMatrix

    def findDistortion(self):
        if not self.result:
            self.calibrateCamera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.result
        print(distCoeffs)
        return distCoeffs

    def findExtrinsicMatrix(self):
        if not self.result:
            self.calibrateCamera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.result

        i = g.extrinsicAttentionImageIndex
        R = np.zeros((3, 3), dtype='float')
        cv2.Rodrigues(rvecs[i], R)
        t = np.array(tvecs[i])
        # print(R)
        # print(t)
        print(np.concatenate((R, t), axis=1))

    def undistort(self):
        if not self.result:
            self.calibrateCamera()
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.result
        for file in g.fileList:
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)
            dest = cv2.undistort(img, cameraMatrix, distCoeffs)
            width, height = img.shape[0], img.shape[1]
            # width : height = 100 : newHeight
            height = int(height * 750 / width)
            width = 750
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_NEAREST)
            dest = cv2.resize(dest, (width, height),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Distorted image", img)
            cv2.imshow("Undistored image", dest)
            cv2.waitKey(500)

class Projector:
    def __init__(self):
        self.calibration = Calibration()

    def projectWords(self, lib):
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = self.calibration.calibrateCamera()

        # load open cv alphabet lib
        fs = cv2.FileStorage(lib, cv2.FILE_STORAGE_READ)

        # generate objectPoints
        objectPoints = []
        shiftX = 7
        shiftY = 5
        for (j, token) in enumerate(g.projectWordInput):
            k = 0
            data = fs.getNode(token).mat().reshape(-1)
            while k < len(data):
                objectPoints.append([
                    data[k+1] + shiftY,
                    data[k] + shiftX,
                    -data[k+2]
                ])
                k += 3
            shiftX -= 3
            if j == 2:
                shiftX = 7
                shiftY -= 3

        # for every image
        for (j, file) in enumerate(g.fileList):
            # get project points
            imagePoints, jacobian = cv2.projectPoints(np.array(
                objectPoints, dtype='float'), rvecs[j], tvecs[j], cameraMatrix, distCoeffs)

            # load original image
            img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1)

            # draw lines
            i = 0
            while i < len(imagePoints):
                cv2.line(img,
                         (int(imagePoints[i][0][0]),
                          int(imagePoints[i][0][1])),
                         (int(imagePoints[i+1][0][0]),
                          int(imagePoints[i+1][0][1])),
                         (0, 0, 255), 10)
                i += 2

            # resize
            width, height = img.shape[0], img.shape[1]
            # width : height = 100 : newHeight
            height = int(height * 750 / width)
            width = 750
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_NEAREST)

            cv2.imshow("projection", img)
            cv2.waitKey(1000)

    def projectWords2D(self):
        self.projectWords(
            "./alphabet_lib_onboard.txt")

    def projectWords3D(self):
        self.projectWords(
            "./alphabet_lib_vertical.txt")


class Window:
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html

    def __init__(self):
        self.windowHeight = 320
        self.UnitWIDTH = 250
        self.UnitWIDTHWithSpace = self.UnitWIDTH+10

        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle('2022 CvDl Hw1')
        self.window.resize(self.UnitWIDTHWithSpace*4, self.windowHeight)
        self.calibration = Calibration()
        self.projector = Projector()
        self.boxA()
        self.boxB()
        self.boxC()
        self.boxD()

    def openFolder(self):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        files.sort(key=lambda x: int(splitext(x)[0]))
        files = [join(folderPath, f) for f in files]
        g.fileList = files
        # reset calibration result
        self.calibration.result = ()
        return files

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self.window)
        btn1.setText('Load Folder')
        btn1.clicked.connect(self.openFolder)

        btn2 = QtWidgets.QPushButton(self.window)
        btn2.setText('Load Image_L')

        btn3 = QtWidgets.QPushButton(self.window)
        btn3.setText('Load Image_R')

        box = QtWidgets.QGroupBox(title="Load Image", parent=self.window)
        box.setGeometry(0, 0, self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)

    def boxB(self):
        btn4 = QtWidgets.QPushButton(self.window)
        btn4.setText('1.1 Find Corners')
        btn4.clicked.connect(self.calibration.findCorders)

        btn5 = QtWidgets.QPushButton(self.window)
        btn5.setText('1.2 Find Intrinsic')
        btn5.clicked.connect(self.calibration.findIntrinsicMatrix)

        comboBox = QtWidgets.QComboBox(self.window)
        comboBox.addItems(['%d' % (i+1,) for i in range(0, 15)])
        comboBox.setGeometry(0, 0, 50, 30)
        comboBox.currentIndexChanged.connect(g.extrinsicAttentionImage)

        btn6 = QtWidgets.QPushButton(self.window)
        btn6.setText('1.3 Find Extrinsic')
        btn6.clicked.connect(self.calibration.findExtrinsicMatrix)

        btn7 = QtWidgets.QPushButton(self.window)
        btn7.setText('1.4 Find Distortion')
        btn7.clicked.connect(self.calibration.findDistortion)

        btn8 = QtWidgets.QPushButton(self.window)
        btn8.setText('1.5 Show Result')
        btn8.clicked.connect(self.calibration.undistort)

        box = QtWidgets.QGroupBox(title="1. Calibration", parent=self.window)
        box.setGeometry(self.UnitWIDTHWithSpace, 0,
                        self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn4)
        layout.addWidget(btn5)

        innerBox = QtWidgets.QGroupBox(title="1.3 Find Extrinsic", parent=box)
        innerLayout = QtWidgets.QVBoxLayout(innerBox)
        innerLayout.addWidget(comboBox)
        innerLayout.addWidget(btn6)

        layout.addWidget(innerBox)
        layout.addWidget(btn7)
        layout.addWidget(btn8)

    def boxC(self):
        input = QtWidgets.QLineEdit(self.window)
        input.setGeometry(0, 0, 100, 30)
        input.textChanged.connect(g.listenProjectWordsInputField)

        btn9 = QtWidgets.QPushButton(self.window)
        btn9.setText('2.1 Show Words on Board')
        btn9.clicked.connect(self.projector.projectWords2D)

        btn10 = QtWidgets.QPushButton(self.window)
        btn10.setText('2.2 Show Words Vertically')
        btn10.clicked.connect(self.projector.projectWords3D)

        box = QtWidgets.QGroupBox(
            title="2. Augmented Reality", parent=self.window)
        box.setGeometry(self.UnitWIDTHWithSpace*2, 0,
                        self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(input)
        layout.addWidget(btn9)
        layout.addWidget(btn10)

    def boxD(self):
        btn11 = QtWidgets.QPushButton(self.window)
        btn11.setText('3.1 Stereo Disparity Map')

        box = QtWidgets.QGroupBox(
            title="3. Stereo Disparity Map", parent=self.window)
        box.setGeometry(self.UnitWIDTHWithSpace*3, 0,
                        self.UnitWIDTH, self.windowHeight)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(btn11)

    def render(self):
        self.window.show()
        sys.exit(self.app.exec())


# const
window = Window()
window.render()
