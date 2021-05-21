import cv2
import numpy as np
import func

width = 700
height = 700
ques = 5
choices = 5
ansCorr = [1, 2, 0, 1, 4]

PATH = r"C:\Users\Lenovo\Desktop\New folder\python\openCV\OMR Detection\1.jpg"

img = cv2.imread(PATH)
img = cv2.resize(img, (width, height))

imgContours = img.copy()
imgBiggestContours = img.copy()
imgFinal = img.copy()
# cv2.imshow("Original", img)

# Preprocessing of Image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# Find Contours
contours, heirarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

# find biggest rectangle contour
rectContour = func.rectContours(contours)
biggestContour = func.getCornerPoints(rectContour[0])
secondBiggestContour = func.getCornerPoints(rectContour[1])
# print(biggestContour)
# print(len(biggestContour))

# print(biggestContour.shape)

gradePoints = secondBiggestContour

# Draw the biggest and second biggest contours corner points
if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 30)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 30)

    biggestContour = func.reorder(biggestContour)
    gradePoints = func.reorder(gradePoints)

    # Apply warpPerspective to get Bird eye's view
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarped = cv2.warpPerspective(img, matrix, (width, height))

    gpt1 = np.float32(gradePoints)
    gpt2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    gmatrix = cv2.getPerspectiveTransform(gpt1, gpt2)
    gradeimgWarped = cv2.warpPerspective(img, gmatrix, (325, 150))

    # cv2.imshow("Grade", gradeimgWarped)

    # Apply Threshold to find Marking Points
    # bubbles with no markings => less amount of pixels
    # bubbles which are marked => more amount of pixels

    imgWarpedGray = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgWarpedGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    # Find each individual bubbles and see which are marked and not marked using pixel values
    bubbles = func.splitBubbles(thresh)
    # cv2.imshow("Test", bubbles[2])   # 3rd bubble in 1st row

    # Total bubbles in img => 24

    # Check marked on not marked
    # Non Zero pixels are greater => marked bubble
    # print(cv2.countNonZero(bubbles[1]), cv2.countNonZero(bubbles[2]))

    # Get non zero pixel values of each bubble
    pixelValues = np.zeros((ques, choices))
    countCols = 0
    countRows = 0

    # iterate through all rows and columns
    for i in bubbles:
        totalPixels = cv2.countNonZero(i)
        pixelValues[countRows][countCols] = totalPixels

        countCols+=1
        if countCols == choices:
            countRows+=1
            countCols = 0

    # print(pixelValues)
    # Each Row => Maximum pixel value -> Marked Bubble

    # Find Index Values of Marked Bubbles
    markedIndex = []
    for x in range(ques):
        arr = pixelValues[x]
        # print("arr", arr)
        markedIndexVal = np.where(arr == np.amax(arr))
        # print(markedIndex[0])
        markedIndex.append(markedIndexVal[0][0])

    # print(markedIndex)
    # print(ansCorr)

    # Grade the Pixels by comparing list of original Answers with Marked Answers
    # If answer correct => score = 1
    # If answer wrong => score = 0
    grading = []
    for i in range(ques):
        if ansCorr[i] == markedIndex[i]:
            grading.append(1)
        else:
            grading.append(0)
    # print(grading)

    finalScore = (sum(grading)/ques)*100
    print(finalScore)

    # Display Answers
    imgResult = imgWarped.copy()
    imgResult = func.showAnswers(imgResult, markedIndex, grading, ansCorr, ques, choices)

    imgRawDrawing = np.zeros_like(imgWarped)
    imgRawDrawing = func.showAnswers(imgRawDrawing, markedIndex, grading, ansCorr, ques, choices)

    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarped = cv2.warpPerspective(imgRawDrawing, invMatrix, (width, height))

    # Display the Grade
    imgRawGraded = np.zeros_like(gradeimgWarped)
    cv2.putText(imgRawGraded, str(int(finalScore))+"%", (60, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
    # cv2.imshow("Grade", imgRawGraded)

    ginvmatrix = cv2.getPerspectiveTransform(gpt2, gpt1)
    gradeInvImgWarped = cv2.warpPerspective(imgRawGraded, ginvmatrix, (width, height))

    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarped, 1, 0)
    imgFinal = cv2.addWeighted(imgFinal, 1, gradeInvImgWarped, 1, 0)

# Image Stacking
imgBlank = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContours, imgBiggestContours, imgWarped, thresh],
            [imgResult, imgRawDrawing, imgInvWarped, imgFinal])

imgStacked = func.stackImages(imgArray, 0.3)

cv2.imshow("Final Result", imgFinal)
cv2.imshow("Stacked Images", imgStacked)

k = cv2.waitKey(0)
