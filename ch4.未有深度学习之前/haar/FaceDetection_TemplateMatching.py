import cv2
import numpy as np

# 读取图片，彩色模式
img_color = cv2.imread('faces.jpg',cv2.IMREAD_COLOR)
# 读取图片，灰度模式
img_gray = cv2.imread('faces.jpg',cv2.IMREAD_GRAYSCALE)
# 读取人脸模板图片，灰度模式
template = cv2.imread('face_template1.jpg',cv2.IMREAD_GRAYSCALE)
# 获取模板尺寸
w, h = template.shape[::-1]
# 模板匹配方法数组
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# 遍历匹配方法
for meth in methods:
    # 拷贝图片
    img_color2 = img_color.copy()
    img_gray2 = img_gray.copy()
    # 把字符串转换成代码
    method = eval(meth)
    # 模板匹配
    res = cv2.matchTemplate(img_gray2,template,method)
    # 获取匹配结果的最大、最小值，及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # TM_SQDIFF 和 TM_SQDIFF_NORMED匹配方法：值越小，越相似
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        # 取最小值位置，作为矩形框左上角位置
        top_left = min_loc
    else:
        # 取最大值位置，作为矩形框左上角位置
        top_left = max_loc
    # 根据模板尺寸计算出：矩形框右下角位置
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 画矩形框
    cv2.rectangle(img_color2,top_left, bottom_right, 255, 2)
    # 显示画好矩形框的图片
    cv2.namedWindow(meth, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(meth,img_color2)
    # 等待退出键
    cv2.waitKey(0)
# 销毁显示窗口
cv2.destroyAllWindows()
