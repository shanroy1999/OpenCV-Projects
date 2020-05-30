import cv2
import numpy as np

cap = cv2.VideoCapture(0)
background = 0

# Capture the background for 30s
for i in range(30):
    ret, background = cap.read()

while True:
    ret, frame = cap.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(frame_hsv, lower_red, upper_red)        # Seperate Cloak Part => Except cloak everything is there

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(frame_hsv, lower_red, upper_red)

    final_mask = mask1 + mask2

    #Noise Removal
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

    mask2 = cv2.bitwise_not(final_mask)

    res1 = cv2.bitwise_and(background, background, mask=final_mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)
    final = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Invisibility", final)
    cv2.imshow("Background", background)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask1", mask1)
    cv2.imshow("Mask2", mask2)
    cv2.imshow("Final Mask", final_mask)
    cv2.imshow("Res 1", res1)
    cv2.imshow("Res 2", res2)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()