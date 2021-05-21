import numpy as np
import cv2
import pickle
from tensorflow import keras

width = 640
height = 480
threshold = 0.65
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

model = keras.models.load_model(r'C:\Users\Lenovo\Desktop\New folder\python\openCV\Text Detection\model_trained.h5')

# Convert Image to Grayscale
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Standardize Lighting of the image
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# Normalize Values between 0 and 1 instead of 0 and 255
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

while True:
    success, imgOrig = cap.read()
    img = np.asarray(imgOrig)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = int(model.predict_classes(img))
    probabilityValue = np.amax(predictions)

    if probabilityValue>threshold:
        cv2.putText(imgOrig, str(classIndex) + "      " + str(probabilityValue), (50, 50), font, 1, (0, 0, 255), 1)

    cv2.imshow("Result", imgOrig)

    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
