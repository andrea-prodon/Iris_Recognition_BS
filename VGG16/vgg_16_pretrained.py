from keras.applications import VGG16
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model, optimizers
from keras.layers import Input, Flatten, Dense, Dropout
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import cv2
import numpy as np
from PIL import Image
from skimage.filters.rank import equalize
from skimage.morphology import disk
from torchvision import datasets, transforms
import os
from dataset import IrisDataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.enabled = False


vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze four convolution blocks
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# Make sure you have frozen the correct layers
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

    x = vgg_model.output
    x = Flatten()(x) 
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x) 
    transfer_model = Model(inputs=vgg_model.input, outputs=x)

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.6, patience=8, verbose=1, mode='max', min_lr=5e-5)
checkpoint = ModelCheckpoint('vgg16_finetune.h15', monitor= 'val_accuracy', mode= 'max', save_best_only = True, verbose= 1)


rootpath = "Dataset/CASIA_Iris_interval_norm/"
transforms_train = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.RandomRotation(10.),
                                        transforms.ToTensor()])
train_dataset = IrisDataset(data_set_path=rootpath, transforms=transforms_train, n_samples = 108)
train_data = [train_dataset[i]['images'] for i in range(len(train_dataset))]

transforms_test = transforms.Compose([transforms.Resize((360, 80)),
                                        transforms.ToTensor()])
test_dataset = IrisDataset(data_set_path=rootpath, transforms=transforms_test, train=False, n_samples=108)
test_data = [test_dataset[i]['images'] for i in range(len(test_dataset))]

learning_rate= 5e-5
transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
history = transfer_model.fit(train_data, test_data, batch_size = 1, epochs=50, validation_data=(train_data,test_data), callbacks=[lr_reduce,checkpoint])


for layer in vgg_model.layers[:15]:
    layer.trainable = False
    x = vgg_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) 
    x = Dense(256, activation='relu')(x)
    x = Dense(8, activation='softmax')(x) 
    transfer_model = Model(inputs=vgg_model.input, outputs=x)

for i, layer in enumerate(transfer_model.layers):
    print(i, layer.name, layer.trainable)

#Augment images
train_datagen = ImageDataGenerator(zoom_range=0.2, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2)
#Fit augmentation to training images
train_generator = train_datagen.flow(X_train,y_train,batch_size=1)
#Compile model
transfer_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#Fit model
history = transfer_model.fit_generator(train_generator, validation_data=(X_test,y_test), epochs=100, shuffle=True, callbacks=[lr_reduce],verbose=1)