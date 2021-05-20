import numpy as np
import cv2
import pickle
from tensorflow import keras

width = 640
height = 480
brightness = 100
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, brightness)

model = keras.models.load_model(r'C:\Users\Lenovo\Desktop\New folder\python\openCV\Traffic Sign Classification\model_trained.h5')

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

def getClassName(classNo):
    if classNo==0: return 'Speed limit 20km/h'
    if classNo==1: return 'Speed limit 30km/h'
    if classNo==2: return 'Speed limit 50km/h'
    if classNo==3: return 'Speed limit 60km/h'
    if classNo==4: return 'Speed limit 70km/h'
    if classNo==5: return 'Speed limit 80km/h'
    if classNo==6: return 'End of Speed limit 80km/h'
    if classNo==7: return 'Speed limit 100km/h'
    if classNo==8: return 'Speed limit 120km/h'
    if classNo==9: return 'No Passing'
    if classNo==10: return 'No Passing for Vehicles over 3.5 metric tons'
    if classNo==11: return 'Right of way at next intersection'
    if classNo==12: return 'Priority Road'
    if classNo==13: return 'Yield'
    if classNo==14: return 'Stop'
    if classNo==15: return 'No Vehicles'
    if classNo==16: return 'Vehicles over 3.5 metric tons prohibited'
    if classNo==17: return 'No Entry'
    if classNo==18: return 'General Caution'
    if classNo==19: return 'Dangerous Curve to the Left'
    if classNo==20: return 'Dangerous Curve to the Right'
    if classNo==21: return 'Double curve'
    if classNo==22: return 'Bumpy road'
    if classNo==23: return 'Slippery road'
    if classNo==24: return 'Road narrows on the right'
    if classNo==25: return 'Road work'
    if classNo==26: return 'Traffic signals'
    if classNo==27: return 'Pedestrians'
    if classNo==28: return 'Children crossing'
    if classNo==29: return 'Bicycles crossing'
    if classNo==30: return 'Beware of ice/snow'
    if classNo==31: return 'Wild animals crossing'
    if classNo==32: return 'End of all speed and passing limits'
    if classNo==33: return 'Turn right ahead'
    if classNo==34: return 'Turn left ahead'
    if classNo==35: return 'Ahead only'
    if classNo==36: return 'Go straight or right'
    if classNo==37: return 'Go straight or left'
    if classNo==38: return 'Keep right'
    if classNo==39: return 'Keep left'
    if classNo==40: return 'Roundabout mandatory'
    if classNo==41: return 'End of no passing'
    if classNo==42: return 'End of no passing by vechiles over 3.5 metric tons'

while True:
    success, imgOrig = cap.read()
    img = np.asarray(imgOrig)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrig, "Class: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrig, "Probability: ", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    if probabilityValue>threshold:
        cv2.putText(imgOrig, str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrig, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrig)

    k = cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
