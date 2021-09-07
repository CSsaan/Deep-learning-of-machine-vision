import numpy as np
import cv2 as cv
img = cv.imread('../data/home.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)

cv.imshow("SIFT", img)
cv.imwrite('sift_keypoints.jpg',img)
cv.waitKey(0)
cv.destroyAllWindows()

