import cv2
import numpy as np

logo = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Adding Watermark\watermark.jpg")
logo = cv2.resize(logo, (600,300))
logo_height, logo_width, ch_logo = logo.shape

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Adding Watermark\room.jpg")
img_height, img_width, ch_img = img.shape

center_y = int(img_height/2)
center_x = int(img_width/2)
top_y = center_y - int(logo_height/2)
left_x = center_x - int(logo_width/2)
bottom_y = top_y+logo_height
right_x = left_x+logo_width

roi = img[top_y: bottom_y, left_x: right_x]
res = cv2.addWeighted(roi, 1, logo, 0.5, 0)

img[top_y: bottom_y, left_x: right_x] = res

#cv2.circle(img, (right_x, bottom_y), 10, (0, 0, 255), -1)
#cv2.circle(img, (left_x, top_y), 10, (0, 0, 255), -1)
cv2.imshow("LOGO", logo)
cv2.imshow("Image", img)
cv2.imshow("Result", res)
cv2.imwrite(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Adding Watermark\room+watermark.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()