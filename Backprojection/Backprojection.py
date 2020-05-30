import cv2
import numpy as np
from matplotlib import pyplot as plt

orig = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Backprojection\goalkeeper.jpg")
orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

roi = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Backprojection\ground.jpg")
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

hue, saturation, value = cv2.split(roi_hsv)

#for h, s, v in zip(hue, saturation, value):
#    print((h, s, v))

roi_hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([orig_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#for k in kernel:
#    print(k)
mask = cv2.filter2D(mask, -1, kernel)
_,mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)

cv2.imshow("Goalkeeper", orig)
cv2.imshow("Ground", roi)
mask = cv2.resize(mask, (800, 500))
cv2.imshow("Backprojection", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.imshow(roi_hist)
#plt.show()