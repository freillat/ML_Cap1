import tensorflow as tf
import pandas as pd
import numpy as np
# import time
import matplotlib.pyplot as plt
# from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.lite as tflite

data_dir = './dataset/'
batch_size = 32
img_height = 256
img_width = 256
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=SEED,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=SEED,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = 4

def create_cnn_model(input_shape=(256, 256, 3)):
    # Build the model
    model = Sequential()
    model.add(Rescaling(1./255, input_shape=input_shape))
    # Convolutional layers
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the output of the convolutional layers
    model.add(Flatten())
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer='adam'
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()
# cnn_model.summary()

history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

cnn_model.save('model_2025_satellite.keras')

# Conversion

model = models.load_model('model_2025_satellite.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with tf.io.gfile.GFile('model_2025_satellite.tflite', 'wb') as f:
    f.write(tflite_model)