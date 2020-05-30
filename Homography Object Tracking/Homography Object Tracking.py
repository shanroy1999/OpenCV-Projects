import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Homography Object Tracking\book.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (400, 500))
cap = cv2.VideoCapture(0)

orb = cv2.ORB_create()
kp_image, desc_image = orb.detectAndCompute(img, None)
# img = cv2.drawKeypoints(img, keypoint, img)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_gray, desc_gray = orb.detectAndCompute(frame_gray, None)
    # frame_gray = cv2.drawKeypoints(frame_gray, kp_gray, frame_gray)
    desc_image = np.float32(desc_image)
    desc_gray = np.float32(desc_gray)
    matches = flann.knnMatch(desc_image, desc_gray, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    matching = cv2.drawMatches(img, kp_image, frame_gray, kp_gray, good_matches, frame_gray)

    # cv2.imshow("Image", img)
    # cv2.imshow("Frame", frame_gray)
    cv2.imshow("Matches", matching)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()