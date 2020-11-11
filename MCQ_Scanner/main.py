import cv2
import numpy as np
import utils

#set the path
path = "Train/sample scanned sheets_Page_07.jpg"
widthImg = 700
heightImg = 700
ans = [0,0,2,1,4,0,9,0,5,2]

#read image
img = cv2.imread(path)

#resizing the image
img = cv2.resize(img,(widthImg,heightImg))
imgcontours = img.copy()
imgFinal = img.copy()
imgBiggestcontours = img.copy()
#preprocessing
grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#adding blur #kernel size #sigmax
blurImg = cv2.GaussianBlur(grayImg,(5,5),1)
imgCanny = cv2.Canny(blurImg,10,50)

#find all contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgcontours, contours,-1,(0,255,0),2)

#find rectangle
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradepoints = utils.getCornerPoints(rectCon[1])
gradepoints2 = utils.getCornerPoints(rectCon[2])
gradepoints3 = utils.getCornerPoints(rectCon[3])
gradepoints4 = utils.getCornerPoints(rectCon[4])
gradepoints5 = utils.getCornerPoints(rectCon[5])
gradepoints6 = utils.getCornerPoints(rectCon[6])
#print(gradepoints5.shape)

if biggestContour.size != 0 and gradepoints.size != 0:
    cv2.drawContours(imgBiggestcontours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestcontours, gradepoints, -1, (255, 0, 0), 20)
    cv2.drawContours(imgBiggestcontours, gradepoints2, -1, (0, 0, 255), 20)
    cv2.drawContours(imgBiggestcontours, gradepoints3, -1, (255, 165, 0), 20)
    cv2.drawContours(imgBiggestcontours, gradepoints4, -1, (255, 255, 0), 20)
    cv2.drawContours(imgBiggestcontours, gradepoints5, -1, (0, 255, 255), 20)
    cv2.drawContours(imgBiggestcontours, gradepoints6, -1, (255, 0, 255), 20)

biggestContour = utils.reorder(biggestContour)
gradepoints = utils.reorder(gradepoints)
gradepoints = utils.reorder(gradepoints)
gradepoints2 = utils.reorder(gradepoints2)
gradepoints3 = utils.reorder(gradepoints3)
gradepoints4 = utils.reorder(gradepoints4)
gradepoints5 = utils.reorder(gradepoints5)
gradepoints6 = utils.reorder(gradepoints6)

#defines the points
pt1 = np.float32(gradepoints3)
pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
matrix = cv2.getPerspectiveTransform(pt1,pt2)
imgwrapColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

# pt1A = np.float32(gradepoints4)
# pt2A = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
# matrixA = cv2.getPerspectiveTransform(pt1A,pt2A)
# imgwrapColoredA = cv2.warpPerspective(img,matrixA,(widthImg,heightImg))


#apply threshold
imgWrapGray = cv2.cvtColor(imgwrapColored,cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgWrapGray,100,255,cv2.THRESH_BINARY_INV)[1]

boxes = utils.splitBoxes(imgThresh)
#cv2.imshow("Test",boxes[2])
print(cv2.countNonZero(boxes[0]),cv2.countNonZero(boxes[2]))

#getting nonzero pixels values
myPixelsVal = np.zeros((10,10))
countC=0
countR=0
for image in boxes:
    totalPixels = cv2.countNonZero(image)
    myPixelsVal[countR][countC]=totalPixels
    countC+=1
    if (countC==10):countR+=1 ;countC=0
    #print(myPixelsVal)

#finding index values of the spot
myIndex = []
for x in range(0,10):
    arr = myPixelsVal[x]
    arr == np.amax(arr)
    #myIndexVal = np.where(arr==np.amax(arr))
    #print(myIndexVal[0])
    #myIndex.append(myIndexVal[0][0])
#print(arr)

#grading
# grading=[]
# for x in range (0,10):
#     if ans[x]==myIndex[x]:
#         grading.append(1)
#     else: grading.append(0)
# print(grading)
#
# #displaying answers
# imgResults = imgwrapColored.copy()
# imgResults = utils.showAnswers(imgResults,myIndex, grading, ans, 10, 10)
# imgRawDrawing = np.zeros_like(imgwrapColored)
# imgRawDrawing = utils.showAnswers(imgRawDrawing,myIndex, grading, ans, 10, 10)
#
# invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
# imgInvWrap = cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))
#
# imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWrap,1,0)
#
# #use stackin function to create array
# imgBlank = np.zeros_like(img)
imageArray = ([img, grayImg,blurImg,imgCanny],
              [imgcontours,imgBiggestcontours,imgwrapColored,imgThresh])
              # [imgResults,imgRawDrawing,imgInvWrap,imgFinal])
imgStacked = utils.stackImages(imageArray,0.3)


cv2.imshow("Final Result",imgFinal)
cv2.imshow("stack image", imgStacked)
cv2.waitKey(0)