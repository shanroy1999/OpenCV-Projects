import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def mouse_drawing(event, x, y, flags, params):
    print(event)
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left Click")
        print(x, y)
        circles.append((x, y))

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)
circles = []

while True:
    ret, frame = cap.read()

    for centre in circles:
        cv2.circle(frame, centre, 5, (0,0,255), -1)
        #print(centre)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key==ord("c"):
        circles = []

cap.release()
cv2.destroyAllWindows()