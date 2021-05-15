import cv2
import numpy as np

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

def nothing(x):
    pass

# Controlling Threshold for canny edge detection using trackbar
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 640, 240)
cv2.createTrackbar("Threshold1", "Trackbar", 23, 255, nothing)
cv2.createTrackbar("Threshold2", "Trackbar", 20, 255, nothing)
cv2.createTrackbar("Area", "Trackbar", 5000, 30000, nothing)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvail = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvail:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)

                if len(imgArray[x][y].shape)==2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank]*rows
        hor_con = [imgBlank]*rows

        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[x].shape)==2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor
    return ver

def getContour(img, imgContour):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 5)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Trackbar")

        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: "+str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: "+str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5))

    # Canny Edge Detection using Trackbar
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbar")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbar")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=1)

    # Draw the Contours
    getContour(imgDilated, imgContour)

    # Image Stacking
    imgStack = stackImages(0.8, ([img, imgBlur, imgCanny], [imgDilated, imgContour, imgContour]))

    cv2.imshow("Image Stacked", imgStack)

    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
