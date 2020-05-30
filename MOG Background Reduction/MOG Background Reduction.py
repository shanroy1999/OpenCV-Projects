import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\MOG Background Reduction\highway.mp4")
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=True)

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Background Reduction", fgmask)

    k = cv2.waitKey(20)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()