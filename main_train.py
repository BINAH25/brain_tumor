import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection  import train_test_split
from keras.utils import normalize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense

image_dir = 'brain_tumor_dataset/'
no_tumor_immages = os.listdir(image_dir + 'no/')
yes_tumor_immages = os.listdir(image_dir + 'yes/')

dataset = []
labels = []
input_size = 200

for i, image_name in enumerate(no_tumor_immages):
    if(image_name.split('.')[1] == 'jpg'):
        image=cv2.imread(image_dir + 'no/'+ image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((input_size,input_size))
        dataset.append(np.array(image))
        labels.append(0)

for i, image_name in enumerate(yes_tumor_immages):
    if(image_name.split('.')[1] == 'jpg'):
        image=cv2.imread(image_dir + 'yes/'+ image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((input_size,input_size))
        dataset.append(np.array(image))
        labels.append(1)
        
dataset = np.array(dataset)
labels = np.array(labels)

x_train,x_test,y_train,y_test = train_test_split(dataset,labels, test_size=0.2, random_state=0)

x_train = normalize(x_train,axis=1)
x_test = normalize(x_test, axis=1)

#model
model = Sequential()

# Extracts features from input images using convolutional operations.
model.add(Conv2D(32,(3,3), input_shape=(input_size,input_size, 3)))

#Introduces non-linearity to the model, allowing it to learn complex patterns.
model.add(Activation('relu'))

#Reduces the spatial dimensions of the input data, retaining the most important information.
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Prepares the data for input into the fully connected (dense) layers.
model.add(Flatten())
#Learns global patterns and relationships from the features extracted by the convolutional layers.
model.add(Dense(64))
model.add(Activation('relu'))
#Helps prevent overfitting by randomly setting a fraction of input units to zero during training.
model.add(Dropout(0.5))
model.add(Dense(1))
#Scales the output to the range [0, 1], providing a probability score for binary classification.
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)

model.save('brain_tumor_10_epochs.h5')