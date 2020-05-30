import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

sound = pyglet.media.load(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Gaze Controlled Keyboard\Bottle Cork.wav", streaming=False)
left = pyglet.media.load(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Gaze Controlled Keyboard\Left.wav", streaming=False)
right = pyglet.media.load(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Gaze Controlled Keyboard\Right.wav", streaming=False)

cap = cv2.VideoCapture(0)
board = np.zeros((300, 1200), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Landmarks Detection\shape_predictor_68_face_landmarks.dat")

keyboard = np.zeros((600,1000,3), np.uint8)

keys_set_1 = {0:'Q',1:'W',2:'E',3:'R',4:'T',
            5:'A',6:'S',7:'D',8:'F',9:'G',
            10:'Z',11:'X',12:'C',13:'V',14:'B'}

keys_set_2 = {0:'Y',1:'U',2:'I',3:'O',4:'P',
            5:'H',6:'J',7:'K',8:'L',9:'_',
            10:'V',11:'B',12:'N',13:'M',14:'<'}

def draw_letters(letter_index, text, letter_light):
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400

    width = 200
    height = 200
    th = 3
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text)/2) + x
    text_y = int((height + height_text)/2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th,y + th), (x + width - th, y + height - th), (255,255,255), -1)
        cv2.putText(keyboard, text, (text_x,text_y), font_letter, font_scale, (51,51,51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th,y + th), (x + width - th, y + height - th), (51,51,51), -1)
        cv2.putText(keyboard, text, (text_x,text_y), font_letter, font_scale, (255,255,255), font_th)

def draw_menu():
    r, c, _ = keyboard.shape
    th_lines = 4
    cv2.line(keyboard, (int(c/2) - int(th_lines/2), 0),(int(c/2) - int(th_lines/2), r),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (80 + int(c/2), 300), font, 6, (255, 255, 255), 5)

def midpoint(p1, p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blink_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    cv2.line(frame, left_point, right_point, (0,0,255), 2)
    cv2.line(frame, center_top, center_bottom, (0,0,255), 2)

    ver_length = hypot((center_top[0] - center_bottom[0]), (center_top[1]-center_bottom[1]))
    hor_length = hypot((left_point[0] - right_point[0]), (left_point[1]-right_point[1]))

    ratio = hor_length/ver_length
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks):

    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    #cv2.polylines(frame, [left_eye_region], True, 255, 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])

    #eye = frame[min_y: max_y, min_x: max_x]
    #eye = cv2.resize(gray_eye, None, fx=8, fy=8)
    #_, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, thresh_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height,width = thresh_eye.shape

    left_thresh = thresh_eye[0:height, 0:int(width/2)]
    left_white = cv2.countNonZero(left_thresh)

    right_thresh = thresh_eye[0:height, int(width/2):width]
    right_white = cv2.countNonZero(right_thresh)

    if(left_white == 0):
        gaze_ratio = 1
    elif(right_white == 0):
        gaze_ratio = 5
    else:
        gaze_ratio = left_white/right_white

    return gaze_ratio

frames = 0
letter_index = 0
blinking_frames = 0
frame_to_blink = 6
frames_active_letter = 9

text = ""
dir_selected = "left"
prev_dir = "left"
select_menu = True
selection_frames = 0

while True:
    ret, frame = cap.read()
    r,c,_ = frame.shape

    keyboard[:] = (26,26,26)
    frames += 1
    direction_frame = np.zeros((500, 500, 3), np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_HERSHEY_PLAIN

    frame[r-50:r, 0:c] = (255,255,255)

    if select_menu is True:
        draw_menu()

    if dir_selected == "left":
        key_set = keys_set_1
    else:
        key_set = keys_set_2
    active_letter = key_set[letter_index]


    play = pyglet.media.Player()
    play.queue(sound)
    play1 = pyglet.media.Player()
    play1.queue(right)
    play2 = pyglet.media.Player()
    play2.queue(left)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x,y), (x1,y1), (0,255,0), 2)

        landmarks = predictor(gray, face)
        left_eye, right_eye = eyes_contour_points(landmarks)

        left_eye_ratio = get_blink_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blink_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)


        if(select_menu is True):
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye)/2

            #cv2.putText(frame, str(left_white), (50,100), font, 2, (0,0,255), 3)
            #cv2.putText(frame, str(right_white), (50,150), font, 2, (0,0,255), 3)
            #cv2.putText(frame, str(gaze_ratio_left_eye), (50,200), font, 2, (0,0,255), 3)
            #cv2.putText(frame, str(gaze_ratio_right_eye), (50,250), font, 2, (0,0,255), 3)

            if(gaze_ratio <= 0.9):
                #cv2.putText(frame, "RIGHT", (50,50), font, 2, (0,0,255), 3)
                #direction_frame[:] = (0,0,255)
                dir_selected="right"
                selection_frames+=1
                if selection_frames==15:
                    select_menu = False
                    play1.play()
                    time.sleep(0.5)
                    frames = 0
                    selection_frames = 0

                if dir_selected != prev_dir:
                    prev_dir = dir_selected
                    selection_frames = 0

            #elif(1 < gaze_ratio < 2):
            #    cv2.putText(frame, "CENTER", (50,50), font, 2, (0,0,255), 3)
            else:
                #direction_frame[:] = (255,0,0)
                #cv2.putText(frame, "LEFT", (50,50), font, 2, (0,0,255), 3)
                dir_selected="left"
                selection_frames+=1
                if selection_frames==18:
                    select_menu = False
                    play2.play()
                    time.sleep(0.5)
                    frames = 0

                if dir_selected != prev_dir:
                    prev_dir = dir_selected
                    selection_frames = 0

        else:
            if(blinking_ratio>5):
                cv2.putText(frame, "BLINKING", (50,150), font, 2, (255,0,0), thickness=3)
                blinking_frames+=1
                frames-=1

                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                if(blinking_frames == frame_to_blink):
                    if active_letter != '_' and active_letter != '<':
                        text+=active_letter
                    if active_letter == '_':
                        text+=" "
                    play.play()
                    select_menu = True
                    time.sleep(0.5)

            else:
                blinking_frames=0

    if(select_menu is False):
        if(frames == frames_active_letter):
            letter_index+=1
            frames = 0

        if(letter_index==15):
            letter_index=0

        for i in range(15):
            if(i==letter_index):
                light=True
            else:
                light=False
            draw_letters(i, key_set[i], light)

    cv2.putText(board, text, (80,100), font, 9, 0, 3)

    percentage_blinking = blinking_frames / frame_to_blink
    loading_x = int(c * percentage_blinking)
    cv2.rectangle(frame, (0, r - 50), (loading_x, r), (51, 51, 51), -1)

    cv2.imshow("Frame",frame)
    #cv2.imshow("Direction",direction_frame)
    cv2.imshow("VIRTUAL KEYBOARD",keyboard)
    cv2.imshow("BOARD",board)

    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
