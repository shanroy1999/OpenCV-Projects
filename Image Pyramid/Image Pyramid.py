import cv2
import numpy as np

img = cv2.imread(r'C:\Users\Lenovo\Desktop\New folder\python\openCV\Image Pyramid\apple.jpg')
c = img.copy()

#Creating Gaussian Pyramid
gp = [c]

for i in range(6):
    c = cv2.pyrDown(c)
    gp.append(c)
    cv2.imshow(str(i), c)

#Creating Laplacian Pyramid
lp = [gp[5]]
for i in range(5,0,-1):
    gp_extended = cv2.pyrUp(gp[i])
    laplacian = cv2.subtract(gp[i-1], gp_extended)
    lp.append(laplacian)
    cv2.imshow(str(i),laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()