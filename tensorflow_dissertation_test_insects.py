# https://medium.com/analytics-vidhya/end-to-end-image-classification-project-using-tensorflow-46e78298fa2f
# https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import gc
import glob

# set the base directory - the current path/folder where this .py program is at
base_dir = os.path.dirname(os.path.realpath(__file__))

# set the train directory, validation directory, test directory
train_dir = os.path.join(base_dir, 'train_insects')
validation_dir = os.path.join(base_dir, 'validation_insects')
test_dir = os.path.join(base_dir, 'test_insects')

# set the data path and get all the files that end with 'g' (i.e. jpg, png, etc...)
data_path = os.path.join(test_dir,'*g')
files = glob.glob(data_path)

# empty list to store the test data
test_data = []

# read and append the image to the list
for image in files:
    img = cv2.imread(image)
    test_data.append(image)

def read_and_process_image(list_of_images):
    X = []
    y = []
    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (150, 150), interpolation=cv2.INTER_CUBIC))
        if 'beetles' in image:
            y.append(1)
        elif 'butterflies' in image:
            y.append(0)
    return X, y
        
# Directory with training butterfly pics
train_butterflies_dir = os.path.join(train_dir, 'butterflies')

# Directory with training beetles pics
train_beetles_dir = os.path.join(train_dir, 'beetles')

# Directory with validation butterfly pics
val_butterflies_dir = os.path.join(validation_dir, 'butterflies')

# Directory with validation beetles pics
val_beetles_dir = os.path.join(validation_dir, 'beetles')

# train the model using the default settings
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation ='sigmoid') ])
    
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# binary classification - to read up more
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# binary classification - to read up more
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 50,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = 25,
    verbose = 2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation loss')
plt.legend()

plt.show()

X_test, y_test = read_and_process_image(test_data)
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1./255)

i = 0
columns = 5
text_labels = []
plt.figure(figsize=(30, 20))

# outputs the prediction on the graph. If it is above 0.5, it is class X, otherwise, it is class Y
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('butterfly')
    else:
        text_labels.append('beetle')
    plt.subplot(5/ columns+1, columns, i+1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
            break

plt.show()
