#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:50:10 2023

@author: chiaweijie
"""


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# hyperparameter
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'vanilla_model.h5'
epochs = 500 
num_classes=7
img_width, img_height = 48, 48
batch_size = 32

# Define the directories where your training and testing images are stored
train_dir = 'emotions/train'
test_dir = 'emotions/test'

# Define the image generators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory( #generator object
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

#Build Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_width, img_height, 1))) 
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

models = [
    ('Adam', keras.optimizers.Adam())
]


train_acc = []
val_acc = []
train_loss = []
val_loss = []
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# Train the models and plot the results
for name, optimizer in models:
    print(f'Training model with {name} optimizer')
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // batch_size, #ensuring that the entire training and validation datasets are used during each epoch of training. 
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_generator.n // batch_size,
                    workers=4,
                    callbacks=[early_stopping,reduce_lr] #can try without scheduler effect also
                    )

    train_acc.append(history.history['accuracy'][-1])
    val_acc.append(history.history['val_accuracy'][-1])
    train_loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])
    
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(['Train', 'Validation'], loc='upper left')
axs[0].grid(False) 

# Plot training and validation loss
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(['Train', 'Validation'], loc='upper left')
axs[1].grid(False) 

# Check if directory exists; if no, create one
model_path = os.path.join(save_dir, model_name)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate_generator(test_generator, steps=test_generator.n // batch_size)
print('Test loss:', scores[0]) #lower better (i.e mistake)
print('Test accuracy:', scores[1]) #higher better

