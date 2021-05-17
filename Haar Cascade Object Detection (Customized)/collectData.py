import cv2
import os
import time

path = "data/images(webcam)"
cameraNo = 0
cameraBrightness = 100
moduleVal = 10
minBlur = 500
greyImage = False
saveData = True
showImage = True
imgWidth = 180
imgHeight = 120

global countFolder
cap = cv2.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, cameraBrightness)

count = 0
countSave = 0

def saveData():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder+=1
    os.makedirs(path + str(countFolder))

if saveData :
    saveData()

while True:
    success, img = cap.read()
    img = cv2.resize(img, (imgWidth, imgHeight))
    if greyImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if count % moduleVal == 0 and blur > minBlur:
            currTime = time.time()
            cv2.imwrite(path+str(countFolder)+'/'+str(countSave)+"_"+str(int(blur))+"_"+str(currTime)+".png", img)
        countSave+=1
    count+=1

    if showImage:
        cv2.imshow("Image", img)

    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
