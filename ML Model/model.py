import torch
import cv2
import numpy as np
import tensorflow as tf
import os

def transform_to_np(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image


labels = []
images = []
input_img_path = 'Dataset/CASIA_Iris_interval_norm'
for (path, dir, files) in os.walk(input_img_path):
    for filename in files:
        fullpath = path + "/" + filename
        frame = cv2.imread(fullpath, cv2.CV_8UC1)
        np_image = transform_to_np(fullpath)
        n_dir = path[path.rindex("/")+1:]
        images.append(np_image)
        labels.append(int(n_dir))
images = np.array(images)
labels = np.array(labels)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(images, labels, epochs=5)

# Save the model
model.save('iris_recognition_model.h5')
