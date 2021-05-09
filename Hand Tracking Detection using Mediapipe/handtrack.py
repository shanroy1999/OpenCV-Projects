import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
run = True
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()

pTime = 0
cTime = 0

while run:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)
    # print(res)
    # print(res.multi_hand_landmarks)

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # if id==0:
                    # cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                # cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
    cv2.imshow("Image",img)
    k = cv2.waitKey(1)

    if k==27:
        run=False
cv2.destroyAllWindows()
