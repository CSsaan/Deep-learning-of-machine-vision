import numpy as np
import cv2 as cv
img = cv.imread('../data/butterfly.jpg',0)

surf = cv.xfeatures2d.SURF_create(400)

#kp, des = surf.detectAndCompute(img,None)
surf.setHessianThreshold(50000)

kp, des = surf.detectAndCompute(img,None)

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imshow('surf',img2)


cv.waitKey(0)
cv.destroyAllWindows()