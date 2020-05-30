import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Eye Motion Tracking\eye_recording.flv")

while True:
    ret, frame = cap.read()
    if ret is False:
        break
    roi = frame[269: 795, 537: 1416]
    rows, cols, _ = roi.shape
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(roi_gray, 5, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 3)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.line(roi, (x+int(w/2), 0), (x+int(w/2), rows), (255, 0, 0), 2)
        cv2.line(roi, (0, y+int(h/2)), (cols, y+int(h/2)), (255, 0, 0), 2)
        break

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Gray", roi_gray)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Gaussian Blur", blur)
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()