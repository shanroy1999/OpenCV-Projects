import cv2
import numpy as np
import time

# YOLO detector - load YOLO
# net = cv2.dnn.readNet(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\yolov3.weights",
#                      r"C:\Users\Lenovo\Desktop\New folder\python\openCV\yolov3.cfg")

# Using Tiny YOLO instead of YOLO for faster detection => less accurate
net = cv2.dnn.readNet(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\yolov3-tiny.weights",
                      r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\yolov3-tiny.cfg")

objects = []
with open(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\YOLO Object Detection\coco.names") as f:
    objects = [line.strip() for line in f.readlines()]

# print(classes)
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(objects), 3))

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\walking.mp4")
start = time.time()
frame_id = 0
while True:
    ret, frame = cap.read()
    frame_id += 1
    height, width, ch = frame.shape

    # Convert original image to blob
    frame_blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)     # True => inverting blue with red
    # print(frame_blob)

    net.setInput(frame_blob)
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
    # print(indexes)
    n_objects = len(boxes)                  # How many boxes detected
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(objects[object_ids[i]])
            confidence = confidences[i]
            color = colors[object_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 20), font, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(round(confidence, 2)), (x, y + 50), font, 1, (0, 0, 0), 2)

    elapsed_time = time.time() - start
    FPS = frame_id / elapsed_time
    cv2.putText(frame, "FPS: "+str(round(FPS, 3)), (10, 50), font, 2, (0, 255, 0), 3)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()