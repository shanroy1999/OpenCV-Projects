import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Image')
cv2.createTrackbar('CE1','Image', 0, 500, nothing)
cv2.createTrackbar('CE2','Image', 0, 500, nothing)

while(1):
    img = cv2.imread(r'C:\Users\Lenovo\Desktop\New folder\python\openCV\Canny Edge Detection\ronaldo.jpg',0)
    pos1 = cv2.getTrackbarPos('CE1', 'Image')
    pos2 = cv2.getTrackbarPos('CE2','Image')
    canny = cv2.Canny(img, pos1, pos2)

    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
    img = cv2.imshow('Image', canny)
cv2.destroyAllWindows()
