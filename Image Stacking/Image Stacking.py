import cv2
import numpy as np

path1 = "apple.jpg"
path2 = "orange.jpg"

img1 = cv2.imread(path1)
# img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2)

img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
img2 = cv2.resize(img2, (0, 0), None, 0.5, 0.5)

# img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

# STACKING IMAGES
# horizontal and vertical stack to join the images
hor = np.hstack((img1, img2))
ver = np.vstack((img1, img2))

cv2.imshow("Horizontal Stack", hor)
cv2.imshow("Vertical Stack", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5, 5), np.uint8)

img = cv2.imread("ronaldo.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=3)
imgEroded = cv2.erode(imgDilation, kernel, iterations=2)

scale = 0.8
img = cv2.resize(img, (0, 0), None, scale, scale)
imgGray = cv2.resize(imgGray, (0, 0), None, scale, scale)
imgBlur = cv2.resize(imgBlur, (0, 0), None, scale, scale)
imgCanny = cv2.resize(imgCanny, (0, 0), None, scale, scale)
imgDilation = cv2.resize(imgDilation, (0, 0), None, scale, scale)
imgEroded = cv2.resize(imgEroded, (0, 0), None, scale, scale)

imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
imgBlur = cv2.cvtColor(imgBlur, cv2.COLOR_GRAY2BGR)
imgCanny = cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2BGR)
imgDilation = cv2.cvtColor(imgDilation, cv2.COLOR_GRAY2BGR)
imgEroded = cv2.cvtColor(imgEroded, cv2.COLOR_GRAY2BGR)

# cv2.imshow("Image", img)
# cv2.imshow("Gray", imgGray)
# cv2.imshow("Blur", imgBlur)
# cv2.imshow("Canny", imgCanny)
# cv2.imshow("Dilation", imgDilation)
# cv2.imshow("Eroded", imgEroded)

hor1 = np.hstack((img, imgGray, imgBlur, ))
hor2 = np.hstack((imgCanny, imgDilation, imgEroded))
ver = np.vstack((hor1, hor2))

cv2.imshow("Stacked Images", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
