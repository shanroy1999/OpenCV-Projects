import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\simpsons.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

copy = img_gray.copy()

template = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Template Matching\barts_face.jpg")
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template_gray.shape[::-1]

# 6 methods of template matching
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = copy.copy()
    method = eval(meth)

    res = cv2.matchTemplate(img, template_gray, method)

    #find the minimum and maximum element values and their positions.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If we use the TM_SQDIFF or TM_SQDIFF_NORMED methods => the minimum value gives best match
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121)
    plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


