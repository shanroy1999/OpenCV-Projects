import cv2
import numpy as np

#Callback Function
def nothing(x):
    pass

cap = cv2.VideoCapture(0)

#Creating a Tracking window with Trackbars to adjust the HSV(Hue, Saturation, Values) values
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)            #For 'Lower Hue' value
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)            #For 'Lower Saturation' value
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)            #For 'Lower Value' value
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)          #For 'Upper Hue' value
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)          #For 'Upper Saturation' value
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)          #For 'Upper Value' value

while(True):
    #img = cv2.imread("Balls.jpg")
    _,img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)          #Convert BGR to HSV

    #Get positions of the trackbars from the Tracking window
    l_h = cv2.getTrackbarPos("LH", "Tracking")          
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")
    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    #Defining Upper Bound and Lower bounds of HSV values - Threshold the hsv values
    l_b = np.array([l_h,l_s,l_v])       #Lower Bound of HSV values
    u_b = np.array([u_h,u_s,u_v])       #Upper Bound of HSV values

    #Creating a mask to filter only the specified color of trackbar using bitwise and operation
    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("frame", img)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()