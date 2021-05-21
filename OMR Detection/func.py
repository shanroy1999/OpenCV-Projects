# Function for Stacking of Images

import cv2
import numpy as np

def stackImages(imgArray, scale, labels=[]):
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

def rectContours(contours):

    rectContour = []
    # filter out according to area of contours
    for i in contours:
        area = cv2.contourArea(i)
        # print("Area: ", area)
        if area > 50:
            # find total length of the contour
            perimeter = cv2.arcLength(i, True)

            # corner count of the contour
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            # print("Corner Points: ", approx)
            # print("Number of Corners: ", len(approx))

            # if number of corners = 4 => rectangle
            if len(approx)==4:
                rectContour.append(i)

    # Sort rectangles based on area
    rectContour = sorted(rectContour, key=cv2.contourArea, reverse=True)

    return rectContour

# find corner points for the biggest rectangle contour
def getCornerPoints(contour):
    perimeter = cv2.arcLength(contour, True)

    # corner count of the contour
    approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)

    return approx

# Reorder the corner points
def reorder(points):
    points = points.reshape((4, 2))

    # origin point => smallest sum
    pointsNew = np.zeros((4, 1, 2), np.int32)

    # print(points)
    add = points.sum(1)
    # print(add)

    pointsNew[0] = points[np.argmin(add)]         # (0, 0)
    pointsNew[3] = points[np.argmax(add)]         # (x, y)

    diff = np.diff(points, axis=1)

    pointsNew[1] = points[np.argmin(diff)]        # (x, 0)
    pointsNew[2] = points[np.argmax(diff)]        # (0, y)

    # print(diff)

    return pointsNew

# Split bubbles
def splitBubbles(img):

    # split horizontally to get all the rows
    rows = np.vsplit(img, 5)

    boxes = []

    # split vertically to get each individual bubble
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)

            cv2.imshow("Bubble", box)

    return boxes

# Display Answers on OMR
def showAnswers(img, markedIndex, grading, ansCorr, ques, choices):
    secWidth = int(img.shape[1] / ques)
    secHeight = int(img.shape[0] / choices)

    for i in range(ques):
        myAns = markedIndex[i]

        # Centre position of marked bubble
        cX = (myAns*secWidth) + secWidth//2
        cY = (i*secHeight) + secHeight//2

        # If correct answer marked => green color
        if grading[i] == 1:
            myColor = (0,255,0)
        else:

            # If wrong answer marked => red color
            myColor = (0, 0, 255)

            # Mark the actual correct answer
            correctAns = ansCorr[i]

            cX_corr = (correctAns*secWidth) + secWidth//2
            cY_corr = (i*secHeight) + secHeight//2

            cv2.circle(img, (cX_corr, cY_corr), 30, (0, 255, 0), cv2.FILLED)

        cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

    return img
