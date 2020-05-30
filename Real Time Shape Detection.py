import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbar")
cv2.createTrackbar("L-H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbar", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbar")
    l_s = cv2.getTrackbarPos("L-S", "Trackbar")
    l_v = cv2.getTrackbarPos("L-V", "Trackbar")
    u_h = cv2.getTrackbarPos("U-H", "Trackbar")
    u_s = cv2.getTrackbarPos("U-S", "Trackbar")
    u_v = cv2.getTrackbarPos("U-V", "Trackbar")

    lower_hsv = np.array([l_h, l_s, l_v])
    upper_hsv = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(frame, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if area > 300:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 3)

        if len(approx) == 4:
            cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 255, 0), 3)
        elif 10 < len(approx) < 20:
            cv2.putText(frame, "Circle", (x, y), font, 1, (0, 255, 0), 3)
        elif len(approx) == 3:
            cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()