# -*-  coding: utf-8 -*-
# Author : Vincent
# Time   : 2018-05-19
# Func   : Laplacian Sharpen

from PIL import Image
import numpy as np

# 读入原图像
img = Image.open('/others/lenna.jpg')
# img.show()

# 为了减少计算的维度，因此将图像转为灰度图
img_gray = img.convert('L')
img_gray.show()

# 得到转换后灰度图的像素矩阵
img_arr = np.array(img_gray)
h = img_arr.shape[0]  # 行
w = img_arr.shape[1]  # 列

# 拉普拉斯算子锐化图像，用二阶微分
new_img_arr = np.zeros((h, w))  # 拉普拉斯锐化后的图像像素矩阵
for i in range(2, h-1):
    for j in range(2, w-1):
        new_img_arr[i][j] = img_arr[i+1, j] + img_arr[i-1, j] + \
                            img_arr[i, j+1] + img_arr[i, j-1] - \
                            4*img_arr[i, j]

# 拉普拉斯锐化后图像和原图像相加
laplace_img_arr = np.zeros((h, w))  # 拉普拉斯锐化图像和原图像相加所得的像素矩阵
for i in range(0, h):
    for j in range(0, w):
        laplace_img_arr[i][j] = new_img_arr[i][j] + img_arr[i][j]

img_laplace = Image.fromarray(np.uint8(new_img_arr))
img_laplace.show()

img_laplace2 = Image.fromarray(np.uint8(laplace_img_arr))
img_laplace2.show()