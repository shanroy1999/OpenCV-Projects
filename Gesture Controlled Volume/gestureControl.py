import cv2
import time
import numpy as np
import handtrack_module as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

width = 640
height = 480
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.handDetector(detectionConf=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol = 0
volBar = 400
volPer = 0
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPos(img, draw=False)

    if len(lmList)!=0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        # Hand Range => 0 => 240
        # Volume Range => -75 => 0

        vol = np.interp(length, [0, 270], [minVol, maxVol])
        volBar = np.interp(length, [0, 270], [400, 150])
        volPer = np.interp(length, [0, 270], [0, 100])
        # print(length, vol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (85,400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85,400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPer)} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 5)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 0), 5)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
