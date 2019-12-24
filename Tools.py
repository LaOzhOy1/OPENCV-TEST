import cv2

import math
import numpy as np
import copy
# 12 月 22 日 测试成功  可以截取部分图像
#  问题一 可能由于拍摄的问题 没法保证 截取出来的 图片 size都相同
#   截取图像 后 因为旋转的不规律性 需要不断的进行左旋
def left_rotate(a,b,c,d):
    while a[0] < c[0] or a[1] < c[1]:
        temp = a
        a = b
        b = c
        c = d
        d = temp
    return a, b , c, d
def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return original_img, gray_img, RedThresh, closed, opened


def findContours_img(original_img, opened):
    _,contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]   # 计算最大轮廓的旋转包围盒
    # print(c)
    rect = cv2.minAreaRect(c)
    # 获取包围盒（中心点，宽高，旋转角度）
    # print(rect)
    box = np.int0(cv2.boxPoints(rect))                           # box
    box[0],box[1],box[2],box[3] = left_rotate(copy.deepcopy(box[0]),copy.deepcopy(box[1]),copy.deepcopy(box[2]),copy.deepcopy(box[3]))
    # print("box[0]:", box[0])
    # print("box[1]:", box[1])
    # print("box[2]:", box[2])
    # print("box[3]:", box[3])
    #
    # print("-----------")
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    return box,draw_img

def Perspective_transform(box,original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
    orignal_H= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    pts2 = np.float32([[int(orignal_W+1),int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+3),int(orignal_H+1)))

    return result_img

def Select_Main(input_dir):
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
    box, draw_img = findContours_img(original_img, opened)
    return Perspective_transform(box, original_img)
    # cv2.imshow("gray", gray_img)
    # cv2.imshow("closed", closed)
    # cv2.imshow("opened", opened)
    # cv2.imshow("draw_img", draw_img)
    # rows, cols = result_img.shape[:2]
    # M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), -90, 1)
    # dst = cv2.warpAffine(result_img, M, (cols, rows))


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def compare_idDiff(orgin,test):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp_origin, des_origin = sift.detectAndCompute(gray_origin, None)  # des是描述子
    # kp_test, des_test = sift.detectAndCompute(gray_test, None)  # des是描述子
    fast = cv2.FastFeatureDetector_create()  # 获取FAST角点探测器
    gray_origin = cv2.cvtColor(orgin, cv2.COLOR_BGR2GRAY) #灰度处理图像
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) #灰度处理图像

    kp_origin = fast.detect(gray_origin, None)
    kp_test = fast.detect(gray_test, None)
    print("Total ORIGIN_Keypoints with nonmaxSuppression: ", len(kp_origin))  # 特征点个数
    print("Total TEST_Keypoints with nonmaxSuppression: ", len(kp_test))  # 特征点个数
    gray_origin = cv2.drawKeypoints(gray_origin, kp_origin, gray_origin, color=(255, 255, 0))  # 画到img上面
    gray_test = cv2.drawKeypoints(gray_test, kp_test, gray_test, color=(255, 255, 0))  # 画到img上面
    # cv2.imshow('',gray_origin)
    # cv2.imshow('test',gray_test)
    if len(kp_test) -len(kp_origin) >50:
        return False
    return True
def isBad(origin_dir,test_dir):
    test_img = Select_Main(test_dir)
    origin_img = Select_Main(origin_dir)
    return compare_idDiff(origin_img,test_img)
