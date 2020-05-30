import cv2
import numpy as np

#Callback function - called whenver trackbar value changes
def nothing(x):                 # x - value of current position of trackbar
    print(x)

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')            #Create window with name 'image'

cv2.createTrackbar('B', 'image', 0, 255, nothing)       #Trackbar for 'B' channel values for 'image' window
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('R', 'image', 0, 255, nothing)

switch = '0 : OFF\n 1 : ON'          #Create a switch with this name
cv2.createTrackbar(switch, 'image', 0, 1, nothing)       #Trackbar for 'switch' in the 'image' window

while(1):
    cv2.imshow('image',img)             #Call the named window and load image inside it
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break
    b = cv2.getTrackbarPos('B', 'image')            #Get the position of trackbar 'B' in window 'image'
    g = cv2.getTrackbarPos('G', 'image')
    r = cv2.getTrackbarPos('R', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s==0:
        img[:]=0                    #Do not change any color if switch=0(OFF)
    else:
        img[:] = [b, g, r]         #Change color only when switch=1(ON) to the current positions of BGR

cv2.destroyAllWindows()