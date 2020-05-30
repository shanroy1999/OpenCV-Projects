import cv2
import numpy as np

# YOLO detector - load YOLO
net = cv2.dnn.readNet(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\yolov3.weights",
                      r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\yolov3.cfg")

objects = []
with open(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\coco.names") as f:
    objects = [line.strip() for line in f.readlines()]

# print(classes)
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(objects), 3))

img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\hall.jpg")
img = cv2.resize(img, None, fx=0.6, fy=0.6)
height, width, ch = img.shape

# Convert original image to blob
img_blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)     # True => inverting blue with red
# print(img_blob)
for blob in img_blob:
    # print(blob)
    for n, img_b in enumerate(blob):
        cv2.imshow(str(n), img_b)                   # Each image => blob of each channel(B, G, R)

net.setInput(img_blob)
outs = net.forward(output_layers)           # Contain all the information about different image classes(objects)
# print(outs)

confidences = []
object_ids = []
boxes = []
for out in outs:
    for detection in out:               # Detect Confidence
        scores = detection[5:]
        object_id = np.argmax(scores)
        confidence = scores[object_id]

        if confidence > 0.4:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), 2)

            y = int(center_x - w / 2)
            x = int(center_y - h / 2)

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            object_ids.append(object_id)
    # print(out)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
n_objects = len(boxes)                  # How many boxes detected
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(objects[object_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 20), font, 1, (0, 0, 0), 2)

cv2.imshow("Room", img)
cv2.waitKey(0)
cv2.destroyAllWindows()