import cv2
import numpy as np
import os

path = "Dataset/CASIA1/7/007_1_1.jpg"
image_read = cv2.imread(path)
output = image_read.copy()
cv2.imshow("original", image_read)

# CANNY

image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)
image_test = cv2.Canny(image_test, 20, 70, apertureSize=3)
cv2.imshow("canny", image_test)

# IRIS CIRCLE

hough_circle = cv2.HoughCircles(image_test, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)

# CANNY 2

image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)
image_test = cv2.Canny(image_test, 100, 120, apertureSize=3)
cv2.imshow("canny-2", image_test)

# PUPIL CIRCLE

circles = cv2.HoughCircles(image_test,cv2.HOUGH_GRADIENT, 1, 800,
                            param1=50, param2=20, minRadius=0, maxRadius=60)
circles = np.round(circles[0, :]).astype("int")
 
for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

cv2.imshow('detected circles',output)
cv2.waitKey(0)