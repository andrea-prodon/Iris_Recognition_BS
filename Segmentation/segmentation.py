import cv2
import numpy as np
import os

path = "C:/Users/giorgia.darmiento/Desktop/UNI/BiometricSystems/Bozza segmentation/CASIA1/2/002_1_1.jpg"
print(path)
image_read = cv2.imread(path)
output = image_read.copy()
image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)
cv2.imshow("imagetest", image_test)
image_test = cv2.Canny(image_test, 80, 100, apertureSize=3)
cv2.imshow("image_test", image_test)

circles = cv2.HoughCircles(image_test,cv2.HOUGH_GRADIENT,1,50,
                            param1=50,param2=50,minRadius=0,maxRadius=0)
circles = np.round(circles[0, :]).astype("int")
 
for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('detected circles',output)
cv2.waitKey(0)