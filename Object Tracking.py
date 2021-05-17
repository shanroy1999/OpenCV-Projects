import cv2
import time

cap = cv2.VideoCapture(0)
tracker = cv2.TrackerMOSSE_create()
tracker = cv2.TrackerCSRT_create()

success, img = cap.read()

# bounding box
bbox = cv2.selectROI("Image", img, False)
tracker.init(img, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(img, "Status : Tracking", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    success, bbox = tracker.update(img)
    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Status : Not detected", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(FPS)), (55, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    k = cv2.waitKey(1)
    if k==27:
        break

cv2.destroyAllWindows()
