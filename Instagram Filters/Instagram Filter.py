import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Landmarks Detection\shape_predictor_68_face_landmarks.dat")

pig_nose = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Instagram Filters\pig nose.png")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        right_nose = (landmarks.part(35).x, landmarks.part(35).y)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1]) * 2)
        nose_height = int(nose_width * 0.77)
        # print(nose_width)
        # cv2.circle(frame, top_nose, 1, (255, 0, 0), 3)

        # cv2.rectangle(frame, (int(center_nose[0] - nose_width/2),
        #              int(center_nose[1] - nose_height/2)),
        #              (int(center_nose[0] + nose_width/2),
        #               int(center_nose[1] + nose_height/2)), (0, 0, 255), 3)

        top_left = (int(center_nose[0] - nose_width/2),
                      int(center_nose[1] - nose_height/2))

        bottom_right = (int(center_nose[0] + nose_width/2),
                       int(center_nose[1] + nose_height/2))

        pig_nose = cv2.resize(pig_nose, (nose_width, nose_height))
        pig_nose_gray = cv2.cvtColor(pig_nose, cv2.COLOR_BGR2GRAY)
        _, nose_mask = cv2.threshold(pig_nose_gray, 25, 255, cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1]+nose_height,
                    top_left[0]: top_left[0]+nose_width]
        
        no_nose_area = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(no_nose_area, pig_nose)

        frame[top_left[1]: top_left[1] + nose_height,
        top_left[0]: top_left[0] + nose_width] = final_nose
        
        cv2.imshow("Nose Area", nose_area)
        cv2.imshow("No nose", no_nose_area)
        cv2.imshow("Nose", pig_nose)
        cv2.imshow("Nose Mask", nose_mask)
        cv2.imshow("Final", final_nose)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()