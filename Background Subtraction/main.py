from os import listdir
from os.path import isfile, join, splitext
from PyQt5 import QtWidgets
import sys
import cv2
import cv2.aruco
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA


class Global:
    videoPath = ""
    imagePath = ""
    fileList = []


g = Global()


class Util:
    def __init__(self):
        pass

    def getMeanStd(frames):
        mean = np.mean(frames, axis=0)
        std = np.std(frames, axis=0)
        # if standard deviation is less then 5, set to 5
        std = np.maximum(std, 5)
        return mean, std

    def backgroundSubtraction():
        # reference:
        # https://shengyu7697.github.io/python-opencv-video/
        if g.videoPath == '':
            print("please choose video")
        cap = cv2.VideoCapture(g.videoPath)
        frames = []
        counter = 25

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("can not receive frame or the video is end")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = gray
            foreground = gray

            if counter > 0:
                frames.append(gray)
            elif counter == 0:
                mean, std = Util.getMeanStd(np.array(frames))
            else:
                diff = np.subtract(np.subtract(gray, mean), std*5)
                mask = np.where(diff > 0, 255, 0)
                mask = mask.astype(dtype=np.uint8)
                mask = mask.reshape(diff.shape+(1,))
                mask = np.concatenate([mask, mask, mask], axis=-1)
                foreground = np.bitwise_and(frame, mask)
            counter -= 1

            cv2.imshow('original video', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('foreground', foreground)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def keypointsOfBlueCircle(gray):
        # 感謝彥甫教練得到媽祖托夢的參數！
        # 找不到媽祖放個佛祖保佑
        #                       _oo0oo_
        #                      o8888888o
        #                      88" . "88
        #                      (| -_- |)
        #                      0\  =  /0
        #                    ___/`---'\___
        #                  .' \\|     |// '.
        #                 / \\|||  :  |||// \
        #                / _||||| -:- |||||- \
        #               |   | \\\  -  /// |   |
        #               | \_|  ''\---/''  |_/ |
        #               \  .-\__  '-'  ___/-. /
        #             ___'. .'  /--.--\  `. .'___
        #          ."" '<  `.___\_<|>_/___.' >' "".
        #         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
        #         \  \ `_.   \_ __\ /__ _/   .-` /  /
        #     =====`-.____`.___ \_____/___.-`___.-'=====
        #                       `=---='
        #
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 90
        params.filterByCircularity = True
        params.minCircularity = .8
        params.filterByConvexity = True
        params.minConvexity = .95
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(gray)

    def detectBlueCircle():
        if g.videoPath == '':
            print("please choose video")
        cap = cv2.VideoCapture(g.videoPath)

        ret, frame = cap.read()
        if not ret:
            print("can not receive frame or the video is end")
            return

        canvas = np.copy(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = Util.keypointsOfBlueCircle(gray)
        for keypoint in keypoints:
            (x, y) = keypoint.pt
            x, y = int(x), int(y)

            canvas = cv2.rectangle(
                canvas, (x-6, y-6), (x+6, y+6), (0, 0, 255), 1)
            canvas = cv2.line(canvas, (x-6, y), (x+6, y), (0, 0, 255), 1)
            canvas = cv2.line(canvas, (x, y-6), (x, y+6), (0, 0, 255), 1)

        cv2.imshow('original frame', frame)
        cv2.imshow('circle detect', canvas)

        cv2.waitKey(0)

    def videoTracking():
        # reference:
        # https://opencv-python-tutorials.readthedocs.io/zh/latest/6.%20%E8%A7%86%E9%A2%91%E5%88%86%E6%9E%90/6.2.%20%E5%85%89%E6%B5%81/
        if g.videoPath == '':
            print("please choose video")
        cap = cv2.VideoCapture(g.videoPath)

        lk_params = {
            'winSize': (31, 31),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03),
            'minEigThreshold': .001
        }

        ret, preFrame = cap.read()
        preGray = cv2.cvtColor(preFrame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(preFrame)

        keypoints = Util.keypointsOfBlueCircle(preGray)

        # for keypoint in keypoints:
        prePts = np.array([
            [[int(keypoint.pt[0]), int(keypoint.pt[1])]] for keypoint in keypoints
        ], dtype=np.float32)

        # Create a mask image for drawing purposes
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("can not receive frame or the video is end")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pts, status, err = cv2.calcOpticalFlowPyrLK(
                preGray, gray, prePts, None, **lk_params)

            # Select good points
            good_new = pts[status == 1]
            good_old = prePts[status == 1]

            # draw the tracks
            for (new, old) in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)),
                                (int(c), int(d)), (0, 255, 255), 2)
            preFrame = frame
            preGray = np.copy(gray)
            prePts = good_new.reshape(-1, 1, 2)

            trajectory = np.fmax(frame, mask)
            cv2.imshow('trajectory', trajectory)

            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def perspectiveTransform():
        if g.videoPath == "":
            print("please choose video")
            return
        if g.imagePath == "":
            print("please choose image")
            return

        cap = cv2.VideoCapture(g.videoPath)
        logo = cv2.imdecode(np.fromfile(g.imagePath, dtype=np.uint8), 1)

        h, w, c = logo.shape
        src = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h],
        ], dtype=np.float32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("can not receive frame or the video is end")
                break
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50))
            frameH, frameW, frameC = frame.shape

            dest = [[0, 0], [0, 0], [0, 0], [0, 0]]
            for (corner, id) in zip(corners, ids):
                corner = np.reshape(corner, (4, 2))
                corner = corner.astype(dtype=np.int)
                # id[0]=1                          id[0]=2
                # corner[0] corner[1]              corner[0] corner[1]
                # corner[2] corner[3]              corner[2] corner[3]

                # id[0]=4                          id[0]=3
                # corner[0] corner[1]              corner[0] corner[1]
                # corner[2] corner[3]              corner[2] corner[3]

                if id[0] >= 1 and id[0] <= 4:
                    # frame = cv2.circle(
                    #     frame, (corner[id[0]-1][0], corner[id[0]-1][1]), 5, (255, 0, 0))
                    dest[id[0]-1] = [corner[id[0]-1][0], corner[id[0]-1][1]]

            if len(dest) < 4:
                continue

            dest = np.array(dest, dtype=np.float32)

            # reference:
            # https://www.wongwonggoods.com/python/python_opencv/opencv-warpperspective/
            # https://steam.oxxostudio.tw/category/python/ai/opencv-mask.html#a7
            rotation, status = cv2.findHomography(src, dest)
            fg = cv2.warpPerspective(
                logo, rotation, (frame.shape[1], frame.shape[0]))

            # create mask in order to merge frame and fg
            mask = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
            mask = np.where(mask > 0, 0, 255)
            mask = np.reshape(mask, mask.shape+(1,))
            mask = np.concatenate([mask, mask, mask],
                                  axis=-1).astype(dtype=np.uint8)

            # merge
            frame = np.bitwise_and(frame, mask)
            frame = np.add(frame, fg)

            cv2.imshow('Perspective Transform', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def reconstruction(images):
        if len(images) == 0:
            return []
        shape = images[0].shape
        res = np.array([
            np.reshape(img, (-1)) for img in images
        ], dtype=np.float32)

        # reference https://ithelp.ithome.com.tw/articles/10206243
        pca = PCA(n_components=27)
        res = pca.fit_transform(res)
        res = pca.inverse_transform(res)

        return [
            cv2.normalize(np.reshape(r, shape), None, 0, 255, cv2.NORM_MINMAX)
            .astype(dtype=np.uint8) for r in res
        ]

    def imageReconstruction():
        if len(g.fileList) == 0:
            print("Please load images")
            return

        images = [
            cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1) for file in g.fileList
        ]

        result = Util.reconstruction(images)

        j = len(images)//2
        for (i, (img, res)) in enumerate(zip(images, result)):
            # print(i+j*(i//j)+1)
            # print(i+j*(i//j+1)+1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            plt.subplot(4, j, i+j*(i//j)+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img)
            plt.subplot(4, j, i+j*(i//j+1)+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(res)
        plt.show()

    def computeReconstructionError():
        if len(g.fileList) == 0:
            print("Please load images")
            return

        images = [
            cv2.imdecode(np.fromfile(file, dtype=np.uint8), 1) for file in g.fileList
        ]
        result = Util.reconstruction(images)

        # convert bgr to gray
        imagesGray = np.array([np.reshape(cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY), (-1)) for img in images])
        resultGray = np.array([np.reshape(cv2.cvtColor(
            res, cv2.COLOR_BGR2GRAY), (-1)) for res in result])

        # computing reconstruction error (RE)
        print(np.reshape(imagesGray[0], (350, 350)))
        print(np.reshape(resultGray[0], (350, 350)))
        cv2.imshow("imggray", np.reshape(imagesGray[0], (350, 350)))
        cv2.imshow("resultGray", np.reshape(resultGray[0], (350, 350)))
        error = np.sqrt(np.sum((imagesGray - resultGray) ** 2, axis=1))
        print("reconstruction error:")
        print(error)
        print("max error:", int(np.max(error)))
        print("min error:", int(np.min(error)))


class Window(QtWidgets.QWidget):
    # Reference:
    # https://steam.oxxostudio.tw/category/python/pyqt5/layout-v-h.html
    # https://shengyu7697.github.io/python-pyqt-qfiledialog/
    # https://zhuanlan.zhihu.com/p/75561654

    def __init__(self):
        super().__init__()
        self.setWindowTitle('2022 CvDl Hw1')
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.boxA())
        layout.addWidget(self.boxB())
        layout.addWidget(self.boxC())
        layout.addWidget(self.boxD())
        layout.addWidget(self.boxE())
        self.setLayout(layout)

    def loadVideo(self):
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.videoPath = filename

    def loadImage(self):
        filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
        if filename:
            g.imagePath = filename

    def loadFolder(self):
        folderPath = QtWidgets.QFileDialog.getExistingDirectory()
        files = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        files.sort()
        files = [join(folderPath, f) for f in files]
        g.fileList = files
        return files

    def boxA(self):
        btn1 = QtWidgets.QPushButton(self)
        btn1.setText('Load Video')
        btn1.clicked.connect(self.loadVideo)

        btn2 = QtWidgets.QPushButton(self)
        btn2.setText('Load Image')
        btn2.clicked.connect(self.loadImage)

        btn3 = QtWidgets.QPushButton(self)
        btn3.setText('Load Folder')
        btn3.clicked.connect(self.loadFolder)

        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn1)
        box.addWidget(btn2)
        box.addWidget(btn3)
        return box

    def boxB(self):
        btn4 = QtWidgets.QPushButton(self)
        btn4.setText('1.1 Background Subtraction')
        btn4.clicked.connect(Util.backgroundSubtraction)

        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn4)
        group = QtWidgets.QGroupBox(
            title="1. Background subtraction", parent=self)
        group.setLayout(box)
        return group

    def boxC(self):
        btn5 = QtWidgets.QPushButton(self)
        btn5.setText('2.1 Preprocessing')
        btn5.clicked.connect(Util.detectBlueCircle)

        btn6 = QtWidgets.QPushButton(self)
        btn6.setText('2.2 Video Tracking')
        btn6.clicked.connect(Util.videoTracking)

        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn5)
        box.addWidget(btn6)
        group = QtWidgets.QGroupBox(
            title="2. Optical Flow", parent=self)
        group.setLayout(box)

        return group

    def boxD(self):
        btn7 = QtWidgets.QPushButton(self)
        btn7.setText('3.1 Perspective Transform')
        btn7.clicked.connect(Util.perspectiveTransform)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn7)
        group = QtWidgets.QGroupBox(
            title="3. Perspective Transform", parent=self)
        group.setLayout(box)
        return group

    def boxE(self):
        btn8 = QtWidgets.QPushButton(self)
        btn8.setText('4.1 Image Reconstruction')
        btn8.clicked.connect(Util.imageReconstruction)

        btn9 = QtWidgets.QPushButton(self)
        btn9.setText('4.2 Compute the Reconstruction Error')
        btn9.clicked.connect(Util.computeReconstructionError)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(btn8)
        box.addWidget(btn9)
        group = QtWidgets.QGroupBox(
            title="4. PCA", parent=self)
        group.setLayout(box)
        return group


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
