import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
from keras.regularizers import l2
from tensorflow.keras import layers
import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
from keras.models import load_model

# # Define the directories where your training and testing images are stored
train_dir = 'emotions/train'
test_dir = 'emotions/test'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'vgg_model.h5'

# Define hyperparameters
batch_size = 128
num_classes = 7
epochs = 200
input_shape = (48, 48, 1)
learning_rate=0.0001
dropout_rate=0.5

# Load the fer2013 dataset
train_datagen = ImageDataGenerator(    
    samplewise_center=True,
    samplewise_std_normalization=True,
    brightness_range=(0.8, 1.2),
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    rescale=1./255)

train_generator = train_datagen.flow_from_directory( 
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

val_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

##Build model
main_input = layers.Input([48, 48, 1])
x = layers.BatchNormalization()(main_input)
x = layers.GaussianNoise(0.01)(x)

base_model = VGG16(weights=None, input_tensor=x, include_top=False)

flatten = Flatten()(base_model.output)

fc = Dense(2048, activation='relu',
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            )(flatten)
fc = Dropout(dropout_rate)(fc)
fc = Dense(2048, activation='relu',
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            )(fc)
fc = Dropout(dropout_rate)(fc)

predictions = Dense(num_classes, activation="softmax")(fc)

model = keras.Model(inputs=main_input, outputs=predictions, name='vgg16')

optimizer = keras.optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['categorical_accuracy'])

early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='min', baseline=None,restore_best_weights=True) # stop if validation loss doesn't improve for 3 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4)

print(f'Training model with VGG model')

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.n // batch_size,  # Number of validation steps
    callbacks=[early_stopping,reduce_lr]
)

# Plot training and validation accuracy
# Set style
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(history.history['categorical_accuracy'])
axs[0].plot(history.history['val_categorical_accuracy'])
axs[0].set_title('Model accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training and validation loss
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(['Train', 'Validation'], loc='upper left')


# Check if directory exists; if no, create one
model_path = os.path.join(save_dir, model_name)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model.save(model_path)
print('Saved trained model at %s ' % model_path)

test_loss, test_acc = model.evaluate(val_generator)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# ##generate confusion matrix
# Get the true labels and predicted labels
y_true = []
y_pred = []
for i in range(val_generator.n // batch_size):
    x_val, labels = val_generator.next()
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(model.predict(x_val), axis=1))

# Get the class names from the generator
class_names = list(val_generator.class_indices.keys())
#{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize the confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot the normalized confusion matrix
plt.figure(figsize=(8,8))
sns.heatmap(cm_norm, annot=True, cmap='Blues',xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print the model summary
# Open the file
with open('VGG_Model_Summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

## Create saliency map
# Load the pre-trained Keras model
model = load_model('saved_models/vgg_model.h5')

# Define the list of emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

fig, axs = plt.subplots(3, 7, figsize=(30, 10))
fig.subplots_adjust(hspace=0.1, wspace=0.2)
# Loop over the emotions and generate the saliency maps
for i, emotion in enumerate(emotions):
    # Load an image from the FER2013 dataset and preprocess it
    image_path = f'emotions/saliency/{emotion}/{emotion}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0).astype(np.float32) / 255.

    # Compute the saliency map
    with tf.GradientTape() as tape:
        inputs = tf.Variable(image, dtype=tf.float32)
        outputs = model(inputs)
    grads = tape.gradient(outputs, inputs)
    saliency_map = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()

    # Normalize the saliency map
    saliency_map -= saliency_map.min()
    if saliency_map.max() != 0:
        saliency_map /= saliency_map.max()

    # Superimpose the saliency map on the original image
    image_with_saliency = cv2.cvtColor(cv2.imread(image_path), cv2.COLORMAP_JET)
    
    # Plot the original image and the saliency map
    axs[0, i].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    axs[0, i].axis('off')
    axs[1, i].imshow(saliency_map, cmap='Oranges')
    axs[1, i].axis('off')
    axs[2, i].imshow(saliency_map, cmap='Oranges', alpha=1.0)
    axs[2, i].imshow(image_with_saliency, cmap='gray', alpha=0.5)
    axs[2, i].axis('off')
    
plt.show()
fig.savefig('myplot.png', dpi=300)

