import cv2
import numpy as np

def nothing(x):
    pass

face_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_eye.xml")
glasses_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_smile.xml")

cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.createTrackbar("Scale", "Frame", 11, 25, nothing)
cv2.createTrackbar("Neighbors", "Frame", 0, 20, nothing)

while True:
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scale = cv2.getTrackbarPos("Scale", "Frame")
    neighbours = cv2.getTrackbarPos("Neighbors", "Frame")

    faces = face_cascade.detectMultiScale(frame_gray, scale/10, neighbours)
    print(faces)

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        roi_gray = frame_gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scale/10, neighbours)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

