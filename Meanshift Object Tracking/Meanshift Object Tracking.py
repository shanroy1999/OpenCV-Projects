import cv2
import numpy as np

def getCoord(event, x, y, flags, param):
    if(event == cv2.EVENT_LBUTTONDOWN):
        print(x, y)

video = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Meanshift Object Tracking\mouthwash.avi")
ret, first_frame = video.read()
x = 302
y = 317
width = 100
height = 100

roi = first_frame[y: y+height, x: x+width]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist(roi_hsv, [0], None, [180], [0, 180])            #First channel => Hue => value from 0 to 179
#print(roi_hist)
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
#print(roi_hist)
term_criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cap = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    _, track = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    print(track)
    x, y, w, h = track
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("First Frame", first_frame)
    #cv2.setMouseCallback("First Frame", getCoord)
    roi = cv2.resize(roi, (300, 300))
    cv2.imshow("ROI", roi)
    cv2.imshow("Backprojection", mask)
    
    key = cv2.waitKey(60)
    if key == 27:
        break

cv2.destroyAllWindows()
video.release()