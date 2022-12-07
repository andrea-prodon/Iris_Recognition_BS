import cv2
import numpy as np

# Questa classe fa praticamente la stessa cosa di segmentation ma per 2 immagini
# Ho provato a comparare 2 immagini dello stesso occhio ma
# A me sembra che questo match non funzioni bene per niente


# SEGMENTATION IMAGE 1

path = "Dataset/CASIA1/47/047_1_1.jpg"

image_read = cv2.imread(path)
output = image_read.copy()
image_test = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test = cv2.GaussianBlur(image_test, (7, 7), 1)
image_test = cv2.Canny(image_test, 20, 70, apertureSize=3)
cv2.imshow("canny outer", image_test)

hough_circle = cv2.HoughCircles(image_test, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)
		
image_test1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test1 = cv2.GaussianBlur(image_test1, (7, 7), 1)
image_test1 = cv2.Canny(image_test1, 100, 120, apertureSize=3)
cv2.imshow("canny-2", image_test1)

circles = cv2.HoughCircles(image_test,cv2.HOUGH_GRADIENT,1,800,
                            param1=50,param2=20,minRadius=0,maxRadius=60)
circles = np.round(circles[0, :]).astype("int")
 
for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

cv2.imshow('detected circles',output)
cv2.waitKey(0)


# SEGMENTATION IMAGE 2

path = "Dataset/CASIA1/47/047_1_2.jpg"

image_read = cv2.imread(path)
output = image_read.copy()
image_test2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test2 = cv2.GaussianBlur(image_test2, (7, 7), 1)
image_test2 = cv2.Canny(image_test2, 20, 70, apertureSize=3)

hough_circle = cv2.HoughCircles(image_test, cv2.HOUGH_GRADIENT, 1.3, 800)
if hough_circle is not None:
    hough_circle = np.round(hough_circle[0, :]).astype("int")
    for (x, y, radius) in hough_circle:
        cv2.circle(output, (x, y), radius, (255, 0, 0), 4)
		
image_test3 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image_test3 = cv2.GaussianBlur(image_test3, (7, 7), 1)
image_test3 = cv2.Canny(image_test3, 100, 120, apertureSize=3)

circles = cv2.HoughCircles(image_test,cv2.HOUGH_GRADIENT,1,800,
                            param1=50,param2=20,minRadius=0,maxRadius=60)
circles = np.round(circles[0, :]).astype("int")
 
for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 2)
		cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)


# MATCH

original = image_test2
image_to_compare = image_test

sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)


# PRINT RESULTS

print("Keypoints 1ST Image: " + str(len(kp_1)))
print("Keypoints 2ND Image: " + str(len(kp_2)))
print("GOOD Matches:", len(good_points))
res = len(good_points) / number_keypoints * 100
print("How good it's the match: ", res)
threshold = 90
if res >= threshold:
    print("Access Granted")
else:
    print("Access Denied")

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

cv2.imshow("result", result)
cv2.imshow("Original", original)
cv2.imshow("Test Input", image_to_compare)

cv2.waitKey(0)
cv2.destroyAllWindows()