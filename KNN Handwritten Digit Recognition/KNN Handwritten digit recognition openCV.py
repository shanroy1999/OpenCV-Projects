import cv2
import numpy as np

digits = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\KNN Handwritten Digit Recognition\digits.png", cv2.IMREAD_GRAYSCALE)
test_set = cv2.imread(r"C:\Users\Lenovo\Desktop\New folder\python\openCV\KNN Handwritten Digit Recognition\test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits, 50)
cells = []

for row in rows:
    row_cells = np.hsplit(row, 50)
    #cv2.imshow("Row Cell", row_cells[0])
    #break
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
        
cells = np.array(cells, dtype=np.float32)
#print(cells)
k = np.arange(10)
print(k)
labels = np.repeat(k, 250)          #Repeat each element in k 250 times
print(labels)

test_set = np.vsplit(test_set, 50)
test_cells = []
for d in test_set:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32)

#print(test_cells)

knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)
print(result)

cv2.imshow("First row", rows[0])
cv2.imshow("Digits", digits)
cv2.waitKey(0)
cv2.destroyAllWindows()