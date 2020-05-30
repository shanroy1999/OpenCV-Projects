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
cv2.createTrackbar("U - S", "Trackbar", 255, 179, nothing)
cv2.createTrackbar("U - V", "Trackbar", 255, 179, nothing)

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbar")
    l_s = cv2.getTrackbarPos("L - S", "Trackbar")
    l_v = cv2.getTrackbarPos("L - V", "Trackbar")
    u_h = cv2.getTrackbarPos("U - H", "Trackbar")
    u_s = cv2.getTrackbarPos("U - S", "Trackbar")
    u_v = cv2.getTrackbarPos("U - V", "Trackbar")

    lower_hsv = np.array([l_h, l_s, l_v])
    upper_hsv = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(frame_hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
