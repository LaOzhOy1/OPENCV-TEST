
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5 import uic
from test import *
from Tools import *
import sys
import tkinter as tk
from tkinter import filedialog
import os
import cv2
# import cv2
form_class = uic.loadUiType('test.ui')[0]


class MyTestClass(QtWidgets.QMainWindow,form_class):


    def __init__(self,parent=None):
        QtWidgets.QMainWindow.__init__(self,parent)
        self.setupUi(self)
        self.__origin_img= None
        self.__test_img = None
    def get_origin_img(self):
        return self.__origin_img
    def set_origin_img(self,__origin_img):
        self.__origin_img = __origin_img
    def get_test_img(self):
        return self.__test_img
    def set_test_img(self, __test_img):
        self.__test_img = __test_img

    def load_origin_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.bmp;;*.jpg;;*.png;;All Files(*)")
        img = cv2.imread(imgName)
        image_show = QPixmap(imgName).scaled(self.origin_img.width(), self.origin_img.height())
        self.origin_img.setStyleSheet("border: 2px solid red")
        self.origin_img.setPixmap(image_show)
        self.__origin_img = imgName

    def load_test_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.bmp;;*.jpg;;*.png;;All Files(*)")

        img = cv2.imread(imgName)
        self.__test_img = imgName

        image_show = QPixmap(imgName).scaled(self.test_img.width(), self.test_img.height())
        self.test_img.setStyleSheet("border: 2px solid red")
        self.test_img.setPixmap(image_show)

    def compare_img(self):

        result=isBad(self.__origin_img,self.__test_img)
        if  result is False:
            self.result.append("该测试产品为劣质产品")
        else:
            self.result.append("该测试产品为优质产品")
if __name__ == '__main__':

    # 设置文件对话框会显示的文件类型
    app = QApplication(sys.argv)
    mytestclass = MyTestClass()
    mytestclass.show()
    cv2.waitKey(0)
    sys.exit(app.exec_())
