#Capture Live Screen from camera
import cv2
import datetime
cap = cv2.VideoCapture(0)           #Device index of camera which reads the video or can give the name of video file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))     # For outputing the frame captured

# We can set the properties manually - but camera will take only the resolution available(maximum) for the camera
#cap.set(3, 1280)      #property no. 3 - frame width - new value = 1280
#cap.set(4, 720)       #property no. 4 - frame height - new value = 720

print(cap.isOpened())               # True if the file path is correct or the index is correct, False otherwise
while(True):                        #Capture Frame continuously
  ret, frame = cap.read()           #read() returns true if frame is available, ret = True if frame avail, frame - captures the frame

  #Get the properties of the frame - each property has its own unique property number
  if ret==True:
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     #frame width - property number 3 - can also give property number as argument for get()
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    #frame height
    print(cap.get(cv2.CAP_PROP_FPS))             #frame rate
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))     #Number of frames in the video file
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))      #0-based index of the frame to be decoded/captured next.
    print(cap.get(cv2.CAP_PROP_POS_MSEC))        #Current position of the video file in milliseconds.
    print(cap.get(cv2.CAP_PROP_FOURCC))          #4-character code of video codec
    print(cap.get(cv2.CAP_PROP_EXPOSURE))        #exposure of image
    print(cap.get(cv2.CAP_PROP_BRIGHTNESS))      #brightness of image
    print(cap.get(cv2.CAP_PROP_GAIN))            #gain of image
    print(cap.get(cv2.CAP_PROP_CONTRAST))        #contrast of image
    print(cap.get(cv2.CAP_PROP_SATURATION))      #saturation of image
    print(cap.get(cv2.CAP_PROP_HUE))             #hue of image
    print(cap.get(cv2.CAP_PROP_MODE))            #indicate current capture mode
    print(cap.get(cv2.CAP_PROP_POS_AVI_RATIO))
    print(cap.get(cv2.CAP_GSTREAMER))
    print(cap.get(cv2.CAP_PROP_CONVERT_RGB))
    print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    print(cap.get(cv2.CAP_PROP_XI_DATA_FORMAT))
    print(cap.get(cv2.CAP_PROP_AUTOFOCUS))
    print(cap.get(cv2.CAP_PROP_SHARPNESS))
    print(cap.get(cv2.CAP_PROP_MONOCHROME))
    print(cap.get(cv2.CAP_PROP_GAMMA))
    print(cap.get(cv2.CAP_PROP_TEMPERATURE))
    print(cap.get(cv2.CAP_PROP_TRIGGER))
    print(cap.get(cv2.CAP_PROP_TRIGGER_DELAY))
    print(cap.get(cv2.CAP_PROP_ZOOM))
    print(cap.get(cv2.CAP_PROP_FOCUS))
    print(cap.get(cv2.CAP_PROP_GUID))
    print(cap.get(cv2.CAP_PROP_BACKLIGHT))
    print(cap.get(cv2.CAP_PROP_ISO_SPEED))
    print(cap.get(cv2.CAP_PROP_PAN))
    print(cap.get(cv2.CAP_PROP_TILT))
    print(cap.get(cv2.CAP_PROP_ROLL))
    print(cap.get(cv2.CAP_PROP_IRIS))
    print(cap.get(cv2.CAP_PROP_SETTINGS))
    print(cap.get(cv2.CAP_PROP_BUFFERSIZE))
    print(cap.get(cv2.CAP_PROP_BACKEND))
    print(cap.get(cv2.CAP_PROP_CHANNEL))
    print(cap.get(cv2.CAP_PROP_AUTO_WB))

    out.write(frame)
  
  #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        #Convert BGR to grayscale and output it

  #Show width and height on the video
  font = cv2.FONT_HERSHEY_SIMPLEX
  text = 'Width: ' + str(cap.get(3)) + 'Height: ' + str(cap.get(4))
  #frame = cv2.putText(frame, text, (10,50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

  #Show current date and time
  datet = str(datetime.datetime.now())
  frame = cv2.putText(frame, datet, (10,50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)

  cv2.imshow('frame',frame)                     #Show frame inside the window named 'frame'
  if(cv2.waitKey(1) & 0xFF == ord('q')):        #Wait for user input
    break
cap.release()               #release capture variable
out.release()
cv2.destroyAllWindows()