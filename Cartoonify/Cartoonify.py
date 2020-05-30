import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cartoonify\adv.jpeg")
#edges = cv2.Canny(img, 75, 150)
#blur = cv2.medianBlur(img, 5)
#blur_edge = cv2.Canny(blur, 75, 150)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

#color = cv2.medianBlur(img, 21)
color = cv2.bilateralFilter(img, 9, 500, 500)
cartoon = cv2.bitwise_and(color, color, mask=thresh)

cv2.imshow("adv",img)
#cv2.imshow("Gray",gray)
#cv2.imshow("Blurred",blur)
#cv2.imshow("Edges",edges)
#cv2.imshow("Blur Edges",blur_edge)
#cv2.imshow("Threshold",thresh)
#cv2.imshow("color",color)
cv2.imshow("Cartoon",cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()