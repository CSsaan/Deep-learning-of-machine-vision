import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/others/opencv.png',0) #直接读为灰度图像
for i in range(2000): #添加点噪声
    temp_x = np.random.randint(0,img.shape[0])
    temp_y = np.random.randint(0,img.shape[1])
    img[temp_x][temp_y] = 255

blur_1 = cv2.GaussianBlur(img,(5,5),0)

blur_2 = cv2.medianBlur(img,5)

plt.subplot(1,3,1),plt.imshow(img,'gray')#默认彩色，另一种彩色bgr
plt.subplot(1,3,2),plt.imshow(blur_1,'gray')
plt.subplot(1,3,3),plt.imshow(blur_2,'gray')
plt.show()