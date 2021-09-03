from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import threading
import importlib
import queue
import asyncio
import cv2
import sys
import os
from tire import detection
from tire import detect_dot, detect_wear, detect_defect, detect_text
import skimage
import glob

form_class = uic.loadUiType("hansung.ui")[0]

TREAD = 0
DEFECT = 1
DOT = 2
SIDEWALL = 3
TEXT = 4

class Window(QDialog, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        ### Button Event
        self.btn_reset.clicked.connect(self.btn_reset_clicked)
        self.btn_tread.clicked.connect(self.btn_tread_clicked)
        self.btn_sidewall.clicked.connect(self.btn_sidewall_clicked)

        self.widths = []
        self.heights = []
        self.views = []

        self.widths.append(self.widget_view_1.frameSize().width())
        self.widths.append(self.widget_view_2.frameSize().width())
        self.widths.append(self.widget_view_3.frameSize().width())
        self.widths.append(self.widget_view_4.frameSize().width())
        self.widths.append(self.widget_view_5.frameSize().width())

        self.heights.append(self.widget_view_1.frameSize().height())
        self.heights.append(self.widget_view_2.frameSize().height())
        self.heights.append(self.widget_view_3.frameSize().height())
        self.heights.append(self.widget_view_4.frameSize().height())
        self.heights.append(self.widget_view_5.frameSize().height())

        self.views.append(OwnImageWidget(self.widget_view_1))
        self.views.append(OwnImageWidget(self.widget_view_2))
        self.views.append(OwnImageWidget(self.widget_view_3))
        self.views.append(OwnImageWidget(self.widget_view_4))
        self.views.append(OwnImageWidget(self.widget_view_5))


    def btn_reset_clicked(self):
        self.reset_all_ui()


    def reset_all_ui(self):
        for view in self.views:
            view.image = None

        self.lbl_caution.setStyleSheet('background-color: transparent')
        self.lbl_safe.setStyleSheet('background-color: transparent')
        self.lbl_caution.setText('')
        self.lbl_safe.setText('')
        self.lbl_defect.setText('')
        self.lbl_dot.setText('')
        self.lbl_tread.setText('')
        self.lbl_sidewall.setText('')
        self.lbl_wear.setText('')

        self.update()


    def btn_tread_clicked(self):
        self.views[TREAD].image = None
        self.views[DEFECT].image = None

        self.lbl_caution.setStyleSheet('background-color: transparent')
        self.lbl_safe.setStyleSheet('background-color: transparent')
        self.lbl_caution.setText('')
        self.lbl_safe.setText('')
        self.lbl_defect.setText('')
        self.lbl_tread.setText('')
        self.lbl_wear.setText('')

        self.update()

        try:
            fname = QtWidgets.QFileDialog.getOpenFileName(self)
            image = skimage.io.imread(fname[0])

            # 트레드 영역 크롭
            crop_tread, bool = detection.detect_tire(image, type='tread')

            if bool:  # true
                self.draw(crop_tread, TREAD)
                self.lbl_tread.setText('Cropping 결과')

                # 손상 부위 검출
                result = detect_defect.predict(crop_tread)

                if result is not None:
                    self.draw(result, DEFECT)
                    self.lbl_defect.setText('손상부위 검출 결과')
                else:
                    self.lbl_defect.setText('손상부위 미검출')


                # 마모도 검출
                result = detect_wear.predict(crop_tread)
                self.lbl_wear.setText('마모도 측정 결과')
                if result:
                    self.lbl_caution.setStyleSheet('background-color: red')
                    self.lbl_caution.setText('주의')
                else:
                    self.lbl_safe.setStyleSheet('background-color: skyblue')
                    self.lbl_safe.setText('안전')

            else:
                self.lbl_tread.setText('결과 없음')

        except Exception as e:
            print(e)



    def btn_sidewall_clicked(self):
        self.views[SIDEWALL].image = None
        self.views[DOT].image = None

        self.lbl_dot.setText('')
        self.lbl_sidewall.setText('')

        self.update()

        try:
            fname = QtWidgets.QFileDialog.getOpenFileName(self)
            image = skimage.io.imread(fname[0])

            # 사이드월 영역 크롭
            crop_sidewall, bool = detection.detect_tire(image, type='sidewall')

            if bool:  # true
                self.draw(crop_sidewall, SIDEWALL)
                self.lbl_sidewall.setText('Cropping 결과')

                result = detect_dot.predict(crop_sidewall)

                if result is not None:
                    full_shot = result[0]
                    crop_shot = np.uint8(result[1][0] * 255)

                    self.draw(full_shot, DOT)
                    self.lbl_dot.setText('DOT 영역 검출 결과')

                    files = glob.glob('temp/*')
                    for fn in files:
                        os.remove(fn)

                    skimage.io.imsave("temp/cropped_dot_text.jpg", crop_shot)

                    ocr_result = detect_text.demo(True)

                    self.draw(crop_shot, TEXT)
                    self.lbl_text.setText(f'{ocr_result[0]:10s} ({ocr_result[1]:0.2f} %)')

                else:
                    self.lbl_dot.setText('DOT 영역 미검출')


            else:
                self.lbl_sidewall.setText('결과 없음')

        except Exception as e:
            print(e)


    def draw(self, image, idx):

        img_height, img_width, img_colors = image.shape

        scale_w = float(self.widths[idx]) / float(img_width)
        scale_h = float(self.heights[idx]) / float(img_height)

        scale = min([scale_w, scale_h])

        if scale == 0:
            scale = 1

        img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.views[idx].setImage(image)

    

class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

