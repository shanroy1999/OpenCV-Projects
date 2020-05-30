import cv2
import numpy as np

def getCoord(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:          #if we click left button down
        print(x, y)

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Camshift Object Tracking\holding book.jpeg")
img = cv2.resize(img, (400, 500))
roi = img[201:499, 113:399]
x = 113
y = 252
width = 399-x
height = 499-y
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
term_criteria = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.resize(frame_hsv, (400, 500))
    mask = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
    ret, track = cv2.CamShift(mask, (x, y, width, height), term_criteria)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)

    cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Backprojection", mask)
    key = cv2.waitKey(1)
    if key == 27:
        break

#cv2.imshow("Book", img)
#cv2.imshow("ROI", roi)
#cv2.setMouseCallback('Book', getCoord)
#cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()