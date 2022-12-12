# model

import cv2
import numpy as np

image = cv2.imread('Dataset/CASIA_Iris_interval_norm/1/001_1_1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (224, 224))

image = image.astype(np.float32) / 255.0

image = np.expand_dims(image, axis=-1)

print(image.shape)
