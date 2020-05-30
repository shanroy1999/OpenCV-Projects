import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Messi.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

# top left corner(x), top left corner(y), width, height
rect = (166, 32, 606, 622)

# grabCut(image, mask, rectangle, backgroundmask, foregroundmask, no. of iteration algo runs, )
cv2.grabCut(img,mask,rect,bgModel,fgModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
# Modify mask such that all 0-pixel and 2-pixel are put to 0(background) and the 1-pixel and 3-pixel are put to 1(foreground)

cut = img*mask2[:, :, np.newaxis]   # Multiply final mask with input image to get segmented image

plt.imshow(img)
plt.show()
plt.imshow(cut)
plt.colorbar()
plt.show()
