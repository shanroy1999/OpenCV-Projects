import cv2
import numpy as np

#mouse callback function
def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(255,0,0),-1)            #Create a circle(point)
        pts.append((x,y))                                 #Append the coordinates of the point clicked in the list
        if(len(pts)>=2):                                #If number of times clicked>=2, then join the circles with line
            cv2.line(img, pts[-1], pts[-2], (255,0,0), 5)       
        cv2.imshow('image',img)

# Create a black image, a window and bind the function to window
img = np.zeros((1024,1024,3), np.uint8)
pts = []
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0) & 0xFF==ord('q')
cv2.destroyAllWindows()