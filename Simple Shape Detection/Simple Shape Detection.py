import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Simple Shape Detection\shapes.jpg", 0)
_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
font = cv2.FONT_HERSHEY_COMPLEX
 
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(img, [approx], 0, (0), 5)
    # print(len(approx))
    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx) == 3:
        # print(approx)
        # Extract the points
        # print(approx.ravel())
        cv2.putText(img, "Triangle", (x, y), font, 1, (0, 0, 255))
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), font, 1, (0, 0, 255))
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), font, 1, (0, 0, 255))
    elif len(approx) == 6:
        cv2.putText(img, "Hexagon", (x, y), font, 1, (0, 0, 255))
    elif 6 < len(approx) < 15:
        cv2.putText(img, "Ellipse", (x, y), font, 1, (0, 0, 255))
    else:
        cv2.putText(img, "Circle", (x, y), font, 1, (0, 0, 255))

cv2.imshow("Shapes", img)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()