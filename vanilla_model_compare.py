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
import numpy as np

# hyperparameters
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

# Build Model
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

# Print the model summary
# Open the file
with open('Vanilla_Model_Summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
models = [
    ('SGD', keras.optimizers.SGD()),
    ('RMSprop', keras.optimizers.RMSprop()),
    ('Adagrad', keras.optimizers.Adagrad()),
    ('Adadelta', keras.optimizers.Adadelta()),
    ('Adam', keras.optimizers.Adam()),
]


train_acc = []
val_acc = []
train_loss = []
val_loss = []
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
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

##plot optimizer comparision
# Set the width of the bars
bar_width = 0.35

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Create the side-by-side bar chart for accuracy
ax1.bar(np.arange(len(train_acc)), train_acc, width=bar_width, label='Train')
ax1.bar(np.arange(len(val_acc)) + bar_width, val_acc, width=bar_width, label='Validation')

# Set the x-axis ticks and labels for the accuracy subplot
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels([name for name, _ in models])

# Set the chart title and legend for the accuracy subplot
ax1.set_title('Accuracy')

# Create the side-by-side bar chart for loss
ax2.bar(np.arange(len(train_loss)), train_loss, width=bar_width, label='Train')
ax2.bar(np.arange(len(val_loss)) + bar_width, val_loss, width=bar_width, label='Validation')

# Set the x-axis ticks and labels for the loss subplot
ax2.set_xticks(np.arange(len(models)))
ax2.set_xticklabels([name for name, _ in models])

# Set the chart title and legend for the loss subplot
ax2.set_title('Loss')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


