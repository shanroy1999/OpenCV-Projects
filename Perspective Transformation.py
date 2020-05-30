import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    _, frame = cap.read()

    cv2.circle(frame, (100, 120), 5, (0, 0, 255), -1)
    cv2.circle(frame, (480, 120), 5, (0, 0, 255), -1)
    cv2.circle(frame, (20, 375), 5, (0, 0, 255), -1)
    cv2.circle(frame, (620, 375), 5, (0, 0, 255), -1)

    pts1 = np.float32([[100, 120], [400, 120], [20, 375], [620, 375]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(frame, matrix, (400, 600))

    cv2.imshow("Frame", frame)
    cv2.imshow("Perspective Transformation", res)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
