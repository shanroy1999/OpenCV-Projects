import cv2
import numpy as np

#Callback function - called whenver trackbar value changes
def nothing(x):                 # x - value of current position of trackbar
    print(x)

cv2.namedWindow('image')            #Create window with name 'image'

cv2.createTrackbar('CP', 'image', 10, 400, nothing)       #Trackbar for 'CP' channel values for 'image' window from 10 to 400

switch = 'color/gray'          #Create a switch with this name
cv2.createTrackbar(switch, 'image', 0, 1, nothing)       #Trackbar for 'switch' in the 'image' window

while(1):
    img = cv2.imread(r'C:\Users\Lenovo\Desktop\New folder\python\openCV\Image Trackbar\ronaldo.jpg')         #Read the image
    pos = cv2.getTrackbarPos('CP', 'image')         #Get the current position of Trackbar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(pos), (50,150), font, 4, (0,0,255))            #Print the position of trackbar

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    s = cv2.getTrackbarPos(switch, 'image')

    if s==0:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        #Change color only when switch=1(ON) to the current positions of BGR
    img = cv2.imshow('image',img)

cv2.destroyAllWindows()