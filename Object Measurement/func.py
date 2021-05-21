# Input colored image to function
# Process the contours

import cv2
import numpy as np

def getContours(img, thresh=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, thresh[0], thresh[1])
    kernel = np.ones((5, 5))
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErode = cv2.erode(imgDilation, kernel, iterations=2)

    if showCanny:
        cv2.imshow("Canny", imgErode)

    contours, heirarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx)==filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    # Sort contours based on area size
    finalContours = sorted(finalContours, key= lambda x: x[1], reverse=True)

    if draw:
        for contour in finalContours:
            cv2.drawContours(img, contour[4], -1, (0, 255, 0), 3)

    return img, finalContours

# reorder the points
def reorder(points):
    # print(points.shape)
    newPoints = np.zeros_like(points)       # Send back in the same shape
    points = points.reshape((4, 2))
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]   # (w, 0)
    newPoints[3] = points[np.argmax(add)]   # (0, h)

    diff = np.diff(points, axis=1)          # differentiation
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]

    return newPoints

# Warp the Image
def warpImg(img, points, width, height, pad=20):
    # print(points)
    points = reorder(points)

    pt1 = np.float32(points)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarped = cv2.warpPerspective(img, matrix, (width, height))
    imgWarped = imgWarped[pad:imgWarped.shape[0]-pad, pad:imgWarped.shape[1]-pad]

    return imgWarped

# Calculate lengths
def findDistance(pts1, pts2):
    return ((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2)**0.5
