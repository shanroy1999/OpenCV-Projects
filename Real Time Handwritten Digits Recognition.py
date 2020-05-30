from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
import cv2
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

model = Sequential()
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)

drawing = False
def draw(event, x, y, flags, params):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img1, (x, y), 10, (255, 0, 0), -1)
            
img1 = np.zeros((512, 512, 1), np.uint8)
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw)

while(1):
    cv2.imshow("Image", img1)
    k = cv2.waitKey(1)
    if k == 27:
        break
    elif k == ord('p'):
        img2 = img1/255.0
        img2 = cv2.resize(img2, (28, 28), cv2.INTER_AREA)
        img2 = img2.reshape(1, 28, 28)

        predict = model.predict_classes(img2)
        print(predict)

    elif k == ord('c'):
        img1 = np.zeros((512, 512, 1), np.uint8)
cv2.destroyAllWindows()