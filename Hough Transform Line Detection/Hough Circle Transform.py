import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Balls 3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)        # Draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)           # Draw Centre of Circle

cv2.imshow("Image",cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
