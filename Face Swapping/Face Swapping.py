import cv2
import numpy as np
import dlib

def show_index(arr):
    index = None
    for num in arr[0]:
        index = num
        break
    return index

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Swapping\keanu.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
w, h, _ = img.shape
mask = np.zeros((w, h), dtype=np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Swapping\shape_predictor_68_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

        # cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    points = np.array(landmarks_points, np.int32)
    hull = cv2.convexHull(points)
    # cv2.polylines(img, [hull], True, (0, 0, 255), 2)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    extracted_face = cv2.bitwise_and(img, img, mask=mask)

    rect = cv2.boundingRect(hull)
    # x, y, w, h = rect
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255))

    # Delaunay Triangulation
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # print(triangles)

    triangle_indices = []

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # print(pt1)
        # cv2.circle(img, pt1, 3, (0, 255, 0), -1)

        # cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        # cv2.line(img, pt2, pt3, (0, 255, 0), 2)
        # cv2.line(img, pt3, pt1, (0, 255, 0), 2)

        index_pt1 = np.where((points == pt1).all(axis=1))
        # print(pt1)
        # print(index_pt1)
        index_pt1 = show_index(index_pt1)
        # print(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        # print(pt2)
        # print(index_pt2)
        index_pt2 = show_index(index_pt2)
        # print(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        # print(pt1)
        # print(index_pt1)
        index_pt3 = show_index(index_pt3)
        # print(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            triangle_indices.append(triangle)

    print(triangle_indices)

# Image 2
img2 = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Face Swapping\tom cruise.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
faces2 = detector(img2_gray)
img2_new_face = np.zeros_like(img2)

for face in faces2:
    landmarks2 = predictor(img2_gray, face)
    landmarks2_points = []
    for n in range(0, 68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks2_points.append((x, y))
        # cv2.circle(img2, (x, y), 2, (0, 0, 255), -1)
        
    p2 = np.array(landmarks2_points, np.int32)
    convexhull2 = cv2.convexHull(p2)

# Delaynay Triangulation on both faces
for triangle_index in triangle_indices:
    # First Face Triangulation
    t1_pt1 = landmarks_points[triangle_index[0]]
    t1_pt2 = landmarks_points[triangle_index[1]]
    t1_pt3 = landmarks_points[triangle_index[2]]
    t1 = np.array([t1_pt1, t1_pt2, t1_pt3], np.int32)
    
    rect1 = cv2.boundingRect(t1)
    x, y, w, h = rect1
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    t1_cropped = img[y: y+h, x: x+w]
    t1_cropped_mask = np.zeros((h, w), np.uint8)
    points1 = np.array([[t1_pt1[0] - x, t1_pt1[1] - y],
                      [t1_pt2[0] - x, t1_pt2[1] - y],
                      [t1_pt3[0] - x, t1_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(t1_cropped_mask, points1, (255, 0, 0))
    t1_cropped = cv2.bitwise_and(t1_cropped, t1_cropped, mask=t1_cropped_mask)

    # cv2.line(img, t1_pt1, t1_pt2, (0, 0, 255), 2)
    # cv2.line(img, t1_pt2, t1_pt3, (0, 0, 255), 2)
    # cv2.line(img, t1_pt1, t1_pt3, (0, 0, 255), 2)

    # Second Face Triangulation
    t2_pt1 = landmarks2_points[triangle_index[0]]
    t2_pt2 = landmarks2_points[triangle_index[1]]
    t2_pt3 = landmarks2_points[triangle_index[2]]
    t2 = np.array([t2_pt1, t2_pt2, t2_pt3], np.int32)

    rect2 = cv2.boundingRect(t2)
    x, y, w, h = rect2
    # cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    t2_cropped = img2[y: y + h, x: x + w]
    t2_cropped_mask = np.zeros((h, w), np.uint8)
    points2 = np.array([[t2_pt1[0] - x, t2_pt1[1] - y],
                       [t2_pt2[0] - x, t2_pt2[1] - y],
                       [t2_pt3[0] - x, t2_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(t2_cropped_mask, points2, (255, 0, 0))
    t2_cropped = cv2.bitwise_and(t2_cropped, t2_cropped, mask=t2_cropped_mask)

    # cv2.line(img2, t2_pt1, t2_pt2, (0, 0, 255), 2)
    # cv2.line(img2, t2_pt2, t2_pt3, (0, 0, 255), 2)
    # cv2.line(img2, t2_pt1, t2_pt3, (0, 0, 255), 2)

    # Warp Triangles
    points1 = np.float32(points1)
    points2 = np.float32(points2)
    matrix = cv2.getAffineTransform(points1, points2)
    # print(matrix)
    warp_triangle = cv2.warpAffine(t1_cropped, matrix, (w, h))
    warp_triangle = cv2.bitwise_and(warp_triangle, warp_triangle, mask=t2_cropped_mask)

    # Reconstruct Second Image
    area_triangle = img2_new_face[y: y + h, x: x + w]
    area_triangle_gray = cv2.cvtColor(area_triangle, cv2.COLOR_BGR2GRAY)
    _, mask_triangles = cv2.threshold(area_triangle_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warp_triangle = cv2.bitwise_and(warp_triangle, warp_triangle, mask=mask_triangles)
    area_triangle = cv2.add(area_triangle, warp_triangle)
    # cv2.imshow("Area of tringle", area_triangle)
    # cv2.waitKey(0)
    img2_new_face[y: y + h, x: x + w] = area_triangle

# Face Swapped(image 1 into image 2)
img2_face_mask = np.zeros_like(img2_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

cv2.imshow("Keanu", img)
cv2.imshow("Tom", img2)
cv2.imshow("Mask", mask)
cv2.imshow("Extracted Face", extracted_face)
cv2.imshow("Cropped Triangle 1", t1_cropped)
cv2.imshow("Triangle 1 Mask", t1_cropped_mask)
cv2.imshow("Cropped Triangle 2", t2_cropped)
cv2.imshow("Triangle 2 Mask", t2_cropped_mask)
cv2.imshow("Warped Triangle", warp_triangle)
cv2.imshow("Image 2 New", img2_new_face)
cv2.imshow("Result", result)
cv2.imshow("Final", seamlessclone)
cv2.waitKey(0)
cv2.destroyAllWindows()