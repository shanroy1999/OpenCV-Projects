import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Motion Detection in Video\random people walk studies.mp4")

#Read 2 frames from the video
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)                #find absolute difference between first and second frame

    #Convert diff to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)           #Contours - easier to find in grayscale mode compared to BGR
    blur = cv2.GaussianBlur(gray, (5,5), 0)           #apply gaussian blur to the image
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)      #apply threshold
    dilate = cv2.dilate(thresh, None, iterations=3)
    contours,_ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    #Create contours
    
    #draw = cv2.drawContours(frame1, contours, -1, (0,255,0), 3)

    #Drawing rectangles around moving persons
    for i in contours:
        (x,y,w,h) = cv2.boundingRect(i)

        if cv2.contourArea(i) < 4000:                #If area of contour is less than 700, do nothing
            continue
        else:
            cv2.rectangle(frame1, (x,y),(x+w,y+h),(0,255,0), 3)
            cv2.putText(frame1, "Status: {} ".format("Movement"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("feed",frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()