import cv2
import numpy as np
import func

webcam = False
# PATH = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Object Measurement\A4.jpg"
PATH = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Object Measurement\A43.jpg"

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 300)

scale = 2

# Width and height of A4 size sheet
width = 210 * scale
height = 297 * scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(PATH)
    img = cv2.resize(img, (0, 0), None, 0.2, 0.2)

    imgContours, cnts = func.getContours(img, draw=True, minArea=50000, filter=4)

    # Corner points for the A4 Paper and warp it
    if len(cnts)!=0:
        biggest = cnts[0][2]
        # print(biggest)
        imgWarped = func.warpImg(img, biggest, width, height)
        cv2.imshow("A4 Paper", imgWarped)

        # Contours of the inner object on the A4 paper
        imgContours2, cnts2 = func.getContours(imgWarped, draw=False, minArea=2000, filter=4, thresh=[50, 50])
        if len(cnts2)!=0:
            for obj in cnts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                newPoints = func.reorder(obj[2])

                newWidth = round((func.findDistance(newPoints[0][0]//scale, newPoints[1][0]//scale)/10), 1)
                newHeight = round((func.findDistance(newPoints[0][0]//scale, newPoints[2][0]//scale)/10), 1)

                cv2.arrowedLine(imgContours2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[1][0][0], newPoints[0][0][1]),
                    (255, 0, 255), 3, 8, 0, 0.05)

                cv2.arrowedLine(imgContours2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[2][0][0], newPoints[2][0][1]),
                    (255, 0, 255), 3, 8, 0, 0.05)

                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(newWidth), (x+30, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                    (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(newHeight), (x-70, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                    (255, 0, 255), 2)

        cv2.imshow("Inner Object", imgContours2)

    cv2.imshow("Original", img)
    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
