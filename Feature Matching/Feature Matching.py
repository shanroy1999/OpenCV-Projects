import cv2
import numpy as np

img1 = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Feature Matching\book.jpg")
img1 = cv2.resize(img1, (400,500))
img2 = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Feature Matching\holding book.jpeg")
img2 = cv2.resize(img2, (400,500))

orb = cv2.ORB_create()
keypoints1, descriptor1 = orb.detectAndCompute(img1, None)
keypoints2, descriptor2 = orb.detectAndCompute(img2, None)

for d in descriptor1:               #Descriptors => array of numbers which describes the features of the image
    print(d)

#Brute Force Matching => compare corresponding descriptors of img1 with that of img2
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)               #Crosscheck = True => Best Match, else all taken
matches = bf.match(descriptor1, descriptor2)
matches = sorted(matches, key = lambda x: x.distance)
print(len(matches))

for m in matches:
    print(m.distance)               #Smaller the distance better the match
    
matching_res = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)
cv2.imshow("Matching",matching_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
