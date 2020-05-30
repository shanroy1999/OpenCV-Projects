import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\simpsons.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\barts_face.jpg")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
r, c = template_gray.shape           #rows=>height, column=>width
w, h = template_gray.shape[::-1]     #Invertion of shape

res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
thresh = 0.8
loc = np.where(res >= thresh)

print(loc)

for pt in zip(*loc[::-1]):
    print(pt)
    cv2.rectangle(img, pt, (pt[0]+w,pt[1]+h), (0,0,255), 3)

cv2.imshow("Image", img)
cv2.imshow("Template", template)
cv2.imshow("Detected", res)
cv2.waitKey(0)

img2 = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\mario.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

template2 = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\mario_template.jpg")
template2_gray = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
r2, c2 = template2_gray.shape           #rows=>height, column=>width
w2, h2 = template2_gray.shape[::-1]     #Invertion of shape

res2 = cv2.matchTemplate(img2_gray, template2_gray, cv2.TM_CCOEFF_NORMED)
thresh2 = 0.8
loc2 = np.where(res2 >= thresh2)

print(loc2)

for pt2 in zip(*loc2[::-1]):
    print(pt2)
    cv2.rectangle(img2, pt2, (pt2[0]+w2, pt2[1]+h2), (0,0,255), 3)

cv2.imshow("Image", img2)
cv2.imshow("Template", template2)
cv2.imshow("Detected", res2)
cv2.waitKey(0)
cv2.destroyAllWindows()