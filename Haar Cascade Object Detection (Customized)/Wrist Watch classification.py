import cv2

""" For creating the custom classifier :

1) Download the software - "Cascade Trainger GUI (version 3.3.1)"
2) Download / capture multiple images for the object which you want to detect / classify. (Example => Image of Wrist Watch)
3) Download / capture other images of random things which are different from the object which you want to detect / classify (Example => Image of bottle / glass / car)
4) Create a new folder "images" and create 2 subdirs "pos" and "neg" inside it
6) Inside the "pos" folder copy the images of the object which you want to detect / classify
7) Inside the "neg" folder copy the images of the random objects which are different from the object which you want to detect / classify
8) Open the software "Cascade Trainger GUI (version 3.3.1)"
9) Go to "Train" tab of the software and set the "Positive Image Percentage" = 100 and "Negative Image Count" = Number of Images in the "neg" folder(should be > "pos" folder)
10) Go to "Common" tab of the software and set the "Number of Stages" = 15, rest everything keep default
11) Go to "Cascade" tab of the software and set the "Sample Width" = 16, rest everything keep default
12) Start the training by clicking the "Start" button of the software
13) Notice in the "images" folder a new subdir will be be created named "classifier" along with 3 other files "neg.lst", "pos.lst" and "pos_samples.vec"
14) Inside the "classifier" subdir use the file "cascade.xml" for classification.

"""

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
