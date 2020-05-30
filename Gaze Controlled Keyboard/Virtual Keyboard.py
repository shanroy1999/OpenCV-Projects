import cv2
import numpy as np

keyboard = np.zeros((450,900,3), np.uint8)

keys_set_1 = {0:'Q',1:'W',2:'E',3:'R',4:'T',5:'Y',
            6:'A',7:'S',8:'D',9:'F',10:'G',11:'H',
            12:'Z',13:'X',14:'C',15:'V',16:'B',17:'N'}

def letters(letter_index, text, letter_light):

    if letter_index==0:
        x = 0
        y = 0
    elif letter_index==1:
        x = 150
        y = 0
    elif letter_index==2:
        x = 300
        y = 0
    elif letter_index==3:
        x = 450
        y = 0
    elif letter_index==4:
        x = 600
        y = 0
    elif letter_index==5:
        x = 750
        y = 0
    elif letter_index==6:
        x = 0
        y = 150
    elif letter_index==7:
        x = 150
        y = 150
    elif letter_index==8:
        x = 300
        y = 150
    elif letter_index==9:
        x = 450
        y = 150
    elif letter_index==10:
        x = 600
        y = 150
    elif letter_index==11:
        x = 750
        y = 150
    elif letter_index==12:
        x = 0
        y = 300
    elif letter_index==13:
        x = 150
        y = 300
    elif letter_index==14:
        x = 300
        y = 300
    elif letter_index==15:
        x = 450
        y = 300
    elif letter_index==16:
        x = 600
        y = 300
    elif letter_index==17:
        x = 750
        y = 300

    width = 150
    height = 150
    th = 3  #thickness

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th,y + th), (x + width - th, y + height - th), (255,255,255), -1)
    else:
        cv2.rectangle(keyboard, (x + th,y + th), (x + width - th, y + height - th), (255,0,0), th)
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 6
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text)/2) + x
    text_y = int((height + height_text)/2) + y
    cv2.putText(keyboard, text, (text_x,text_y), font_letter, font_scale, (0,0,255), font_th)

for i in range(18):
    if(i==5):
        light=True
    else:
        light=False
    letters(i,keys_set_1[i], light)

cv2.imshow("KEYBOARD",keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
