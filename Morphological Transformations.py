import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([150, 150, 50])
    upper_red = np.array([180, 255, 150])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((15, 15), np.float32) / 225
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilate = cv2.dilate(mask, kernel, iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    bh = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel)
    cross = cv2.morphologyEx(mask, cv2.MORPH_CROSS, kernel)
    ellipse = cv2.morphologyEx(mask, cv2.MORPH_ELLIPSE, kernel)
    th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)


    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    cv2.imshow("Erosion", erosion)
    cv2.imshow("Dilation", dilate)
    cv2.imshow("Opening", opening)
    cv2.imshow("Closing", closing)
    cv2.imshow("Gradient", mg)
    cv2.imshow("Blackhat", bh)
    cv2.imshow("Cross", cross)
    cv2.imshow("Elipse", ellipse)
    cv2.imshow("Top Hat", th)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()