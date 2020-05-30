import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Scan QR Code\qr-code.png")
decode = pyzbar.decode(img)

for item in decode:
    print("Data: ", item.data)
cv2.imshow("QR Code", img)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    decode = pyzbar.decode(frame)
    for item in decode:
        cv2.putText(frame, str(item.data), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        #print("Data: ", item.data)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()