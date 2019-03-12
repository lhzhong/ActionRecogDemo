#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import cv2
import time

from PIL import Image
import numpy as np
import operator
from UCFdata import DataSet
from keras.models import load_model

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QMutexLocker, QMutex, QThread, QObject, pyqtSignal


class VideoBox(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()
        self.timer = VideoTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        # self.slot_init()

        # 设置组建与布局
        self.btn_open_camera = QtWidgets.QPushButton("打开相机")
        self.btn_open_camera.setMinimumHeight(50)

        self.btn_select_file = QtWidgets.QPushButton("从文件选")
        self.btn_select_file.setMinimumHeight(50)

        self.btn_close_camera = QtWidgets.QPushButton("退出")
        self.btn_close_camera.setMinimumHeight(50)
        self.btn_close_camera.move(10, 100)

        self.label_show_video = QtWidgets.QLabel()
        self.label_show_video.setFixedSize(641, 481)
        self.label_show_video.setAutoFillBackground(False)

        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 100)

        self.group_result = QtWidgets.QGroupBox("识别结果")
        self.label_result = QtWidgets.QLabel("")

        self.layout_reslut = QtWidgets.QHBoxLayout()
        self.layout_reslut.addWidget(self.label_result)
        self.group_result.setLayout(self.layout_reslut)

        self.layout_fun_button = QtWidgets.QVBoxLayout()
        self.layout_fun_button.addWidget(self.btn_open_camera)
        self.layout_fun_button.addWidget(self.btn_select_file)
        self.layout_fun_button.addWidget(self.btn_close_camera)
        self.layout_fun_button.addWidget(self.label_move)
        self.layout_fun_button.addWidget(self.group_result)

        self.layout_main = QtWidgets.QHBoxLayout()
        self.layout_main.addLayout(self.layout_fun_button)
        self.layout_main.addWidget(self.label_show_video)

        self.setLayout(self.layout_main)
        self.setWindowTitle("demo")
        self.setWindowIcon(QtGui.QIcon("./pic/video.jpeg"))

        self.btn_open_camera.clicked.connect(self.button_open_camera_click)
        self.btn_select_file.clicked.connect(self.button_select_from_file)
        self.btn_close_camera.clicked.connect(self.close)

        self.data = DataSet()
        self.model = load_model('./model/inception.020-1.24.hdf5')

        # 将定时器超时信号和槽函数连接
        self.timer_camera.timeout.connect(self.show_video)
        self.timer.timeSignal.signal.connect(self.predict)

    def predict(self):
        # Load a Tensorflow model into memory.
        if self.cap.isOpened():

            flag, frame = self.cap.read()
            if flag:

                # fps_get = FPS().start()
                model_image_size = (229, 299)
                label_predictions = {}
                for k, label in enumerate(self.data.classes):
                    label_predictions[label] = 0
                    # fps_get.update()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb, mode='RGB')
                resized_image = image.resize(
                    tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
                image_data /= 255.
                image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension.
                predictions = self.model.predict(image_data)

                for k, label in enumerate(self.data.classes):
                    label_predictions[label] = predictions[0][k]

                # if frame_num % 10 == 0:

                sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
                text = "%s: %.2f" % (sorted_lps[0][0], sorted_lps[0][1])
                self.label_result.setText(text)
                print(text)

    def button_open_camera_click(self):
        if not self.timer_camera.isActive():
            flag = self.cap.open(self.CAM_NUM)
            if flag:
                print("Open camera sucessfully.")
                # 设置计时，实时发送超时信号，循环进行
                self.timer_camera.start(10)
                self.btn_open_camera.setText("关闭相机")
                self.timer.frequent = 100
                self.timer.start()
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "请检测相机与电脑是否连接正确",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            self.timer_camera.stop()
            self.timer.stop()
            self.cap.release()
            self.label_show_video.clear()
            self.label_result.clear()
            self.btn_open_camera.setText("打开相机")

    def button_select_from_file(self):
        video_url, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Select Video', '', 'Video Files (*.avi *.mp4)')
        print(video_url)
        # If read failed, consider to install opencv-python
        if video_url != "":

            flag = self.cap.open(video_url)
            if flag:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.timer_camera.start(fps)
                print("Read Video Successful")
                self.timer.frequent = 100
                self.timer.start()
                self.timer.timeSignal.signal.connect(self.predict)
            else:
                print("Read Video failed.")

    def show_video(self):

        if self.cap.isOpened():

            flag, image = self.cap.read()
            if flag:
                show = cv2.resize(image, (640, 480))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
                self.label_show_video.setPixmap(QtGui.QPixmap.fromImage(show_image))
            else:
                self.timer.stop()
                self.cap.release()
                self.label_show_video.clear()
                self.label_result.clear()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "关闭", "是否关闭！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText('确定')
        cacel.setText('取消')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()

        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


class Communicate(QObject):

    signal = pyqtSignal()


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        # self.timeSignal.signal.emit()
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit()
            time.sleep(1)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = VideoBox()
    ui.show()
    sys.exit(app.exec_())
