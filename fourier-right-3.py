import cv2
import numpy as np
import matplotlib.pyplot as plt

# 三、基于透视的图像矫正
# 3.1 直接变换
# 获取图像四个顶点
# 形成变换矩阵
# 透视变换


img = cv2.imread('C:/Users/LaOzhOy1/Pictures/Camera Roll/test.bmp')
H_rows, W_cols= img.shape[:2]
print(H_rows, W_cols)

# 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
pts1 = np.float32([[161, 80], [449, 12], [1, 430], [480, 394]])
pts2 = np.float32([[0, 0],[W_cols,0],[0, H_rows],[H_rows,W_cols],])

# 生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (500,470))

"""
注释代码同效
# img[:, :, ::-1]是将BGR转化为RGB
# plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
# plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# plt.show
"""

cv2.imshow("orgin",img)
cv2.imshow("result",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
