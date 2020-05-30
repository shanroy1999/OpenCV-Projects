import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Cascades\haarcascade_smile.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Recognition and Identification\trainer.yml")

labels = {}
with open(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Recognition and Identification\labels.pkl",'rb') as f:
    im_labels = pickle.load(f)
    labels = {v:k for k, v in im_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = frame_gray[y: y+h, x: x+w]
        roi_color = frame[y: y+h, x: x+w]

        #Identify the face
        id_, conf = recognizer.predict(roi_gray)
        if(conf>=50):
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name, (x,y), font, 1, (0, 0, 255), 3, cv2.LINE_AA)

        img_item = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Images\Shantanu\me11.png"
        cv2.imwrite(img_item, roi_gray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)

        smiles = smile_cascade.detectMultiScale(roi_gray)
        for(sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()