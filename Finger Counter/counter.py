import cv2
import time
import os
import handtrack_module as htm

cap = cv2.VideoCapture(0)

width = 640
height = 480

cap.set(3, width)
cap.set(4, height)

PATH = "images"

myList = os.listdir(PATH)

img_list = []
for img_path in myList:
    image = cv2.imread(f'{PATH}/{img_path}')
    img_list.append(image)

# print(len(img_list))

prevTime = 0

detector = htm.handDetector(detectionConf = 0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPos(img, draw=False)
    # print(lmList)

    if len(lmList)!=0:
        fingers = []

        # For the thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for the fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)

        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = img_list[0].shape
        img[0:h, 0:w] = img_list[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    currTime = time.time()
    FPS = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f'FPS : {int(FPS)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
