import cv2
import numpy as np

#mouse callback function
def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        red = img[x,y,0]
        blue = img[x,y,1]
        green = img[x,y,2]
        cv2.circle(img, (x,y), 3, (0,0,255), -1)
        mycolorimage = np.zeros((512,512,3), np.uint8)
        mycolorimage[:] = [red, blue, green]
        cv2.imshow('color',mycolorimage)

# Create a black image, a window and bind the function to window
img = cv2.imread('ronaldo.jpg')
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0) & 0xFF==ord('q')
cv2.destroyAllWindows()