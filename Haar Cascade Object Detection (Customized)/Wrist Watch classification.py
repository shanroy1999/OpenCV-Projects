import cv2

PATH = "Cascades/haarcascade_watch.xml"
cameraNo = 0
objectName = "Watch"
frameWidth = 640
frameHeight = 480
color = (255, 0, 255)

cap = cv2.VideoCapture(cameraNo)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Capture")
cv2.resizeWindow("Capture", frameWidth, frameHeight+100)
cv2.createTrackbar("Scale", "Capture", 400, 1000, empty)
cv2.createTrackbar("Neighbors", "Capture", 8, 20, empty)
cv2.createTrackbar("Min Area", "Capture", 0, 100000, empty)
cv2.createTrackbar("Brightness", "Capture", 100, 255, empty)

cascade = cv2.CascadeClassifier(PATH)

while True:
    cameraBrightness = cv2.getTrackbarPos("Brighness", "Capture")
    cap.set(10, cameraBrightness)

    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Capture") / 1000)
    neig = cv2.getTrackbarPos("Neighbors", "Capture")
    objects = cascade.detectMultiScale(gray, scaleVal, neig)

    for x, y, w, h in objects:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Capture")
        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), 3)
            cv2.putText(img, objectName, (x, y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 3)
            roi_color = img[y:y+h, x:x+w]

    cv2.imshow("Capture", img)

    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
