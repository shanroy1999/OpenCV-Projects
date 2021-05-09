import cv2
import mediapipe as mp
import time
import handtrack_module as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
run = True
while run:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPos(img, draw=False)
    if len(lmList)!=0:
        print(lmList[4])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
    cv2.imshow("Image",img)
    k = cv2.waitKey(1)

    if k==27:
        run = False
cv2.destroyAllWindows()
