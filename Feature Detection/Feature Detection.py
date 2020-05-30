import cv2

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Feature Detection\cover.jpg", cv2.IMREAD_GRAYSCALE)
#sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create()

#keypoints, descriptors = sift.detectAndCompute(img, None)
#keypoints2, descriptors2 = surf.detectAndCompute(img, None)
keypoints3, descriptors3 = orb.detectAndCompute(img, None)

#img1 = cv2.drawKeypoints(img, keypoints, None)
#img2 = cv2.drawKeypoints(img, keypoints2, None)
img3 = cv2.drawKeypoints(img, keypoints3, None)

cv2.imshow("Cover",img)
#cv2.imshow("SIFT",img1)
#cv2.imshow("SURF",img2)
cv2.imshow("ORB",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()