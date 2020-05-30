import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
for i in events:
    print(i)

def click_event(event, x, y, flags, param):          
    #event - event taking place when clicking, 
    # (x,y) - x and y coordinates on image where we are clicking the mouse
    if event == cv2.EVENT_LBUTTONDOWN:          #if we click left button down
        print(x,', ',y)                          #print x and y coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x,y), font, 1, (255,255,0), 3)
        cv2.imshow('image', img)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y,x,0]                          
        green = img[y,x,1]
        red = img[y,x,2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', ' + str(green) + ', ' + str(red)            #rpint BGR channels(i.e. BGR values)
        cv2.putText(img, strBGR, (x,y), font, 1, (0,255,255), 3)
        cv2.imshow('image', img)

#img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('ronaldo.jpg')
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)        #Call the callback function - click_event(), window name must be same as imshow()
cv2.waitKey(0) & 0xFF==ord('q') 
cv2.destroyAllWindows()




