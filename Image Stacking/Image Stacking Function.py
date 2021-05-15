import cv2
import numpy as np

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

# FOR WEBCAM

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)

    kernel = np.ones((5, 5), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 100, 200)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
    imgEroded = cv2.erode(imgDilation, kernel, iterations=2)

    blankImage = np.zeros((200, 200), np.uint8)

    stackedImage = stackImages(0.5, ([img, imgGray, imgCanny], [imgCanny, imgDilation, imgEroded]))
    # stackedImage = stackImages(0.5, ([img, imgGray, imgCanny], [imgCanny, imgDilation, blankImage]))

    cv2.imshow("Stacked Images", stackedImage)
    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

# FOR IMAGES

# kernel = np.ones((5, 5), np.uint8)
# img = cv2.imread("ronaldo.jpg")
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
# imgCanny = cv2.Canny(imgBlur, 100, 200)
# imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
# imgEroded = cv2.erode(imgDilation, kernel, iterations=2)

# blankImage = np.zeros((200, 200), np.uint8)

# stackedImage = stackImages(0.8, ([img, imgGray, imgCanny], [imgCanny, imgDilation, imgEroded]))
# stackedImage = stackImages(0.8, ([img, imgGray, imgCanny], [imgCanny, imgDilation, blankImage]))

# cv2.imshow("Stacked Images", stackedImage)
# cv2.waitKey(0)
