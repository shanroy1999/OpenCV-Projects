import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Corner Detection\squares.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(img_gray, 5, 0.8, 50)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img, (x,y), 5, (0, 0, 255), -1)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()