import cv2

cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#Method 1
#make_1080p()

#Method 2

def rescale_frame(frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1]*scale_percent/100)
    height = int(frame.shape[0]*scale_percent/100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

while True:
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=30)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

