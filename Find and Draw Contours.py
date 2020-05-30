import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbar")

cv2.createTrackbar("L - H", "Trackbar", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbar", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbar", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbar", 255, 255, nothing)

while True:
    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbar")
    l_s = cv2.getTrackbarPos("L - S", "Trackbar")
    l_v = cv2.getTrackbarPos("L - V", "Trackbar")
    u_h = cv2.getTrackbarPos("U - H", "Trackbar")
    u_s = cv2.getTrackbarPos("U - S", "Trackbar")
    u_v = cv2.getTrackbarPos("U - V", "Trackbar")

    lower_hsv = np.array([l_h, l_s, l_v])
    upper_hsv = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()