import cv2
import matplotlib.pyplot as plt

img = cv2.imread('timg.jpg',0) #直接读为灰度图像
res = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10,10))
cl1 = clahe.apply(img)

plt.subplot(131),plt.imshow(img,'gray')
plt.subplot(132),plt.imshow(res,'gray')
plt.subplot(133),plt.imshow(cl1,'gray')

plt.show()