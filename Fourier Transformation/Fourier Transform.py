import cv2
import numpy as np
import glob

list_letters = glob.iglob(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Fourier Transformation\Letters\*")
#img = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\Letters\a_0.png", cv2.IMREAD_GRAYSCALE)
#print(list_letters)

for letter in list_letters:
    img = cv2.imread(letter, cv2.IMREAD_GRAYSCALE)
    #print(letter)
    fourier = np.fft.fft2(img)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude_spectrum = 20 * np.log(np.abs(fourier_shift))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    img_and_magnitude = np.concatenate((img, magnitude_spectrum), axis=1)
    cv2.imshow(letter, img_and_magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()