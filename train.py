import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import tensorflow.lite as tflite

data_dir = './dataset/'
batch_size = 32
img_height = 256
img_width = 256
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ds_generator = ImageDataGenerator(rescale=1./255,validation_split=0.2)
train_ds = ds_generator.flow_from_directory(
    data_dir,
    seed=SEED,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_ds = ds_generator.flow_from_directory(
    data_dir,
    seed=SEED,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

num_classes = 4

def create_cnn_model(input_shape=(256, 256, 3)):
    # Build the model
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
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
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer='adam'
    # optimizer=SGD(learning_rate=0.001, momentum=0.95)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_cnn_model()

history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

labels = {
    0: 'cloudy',
    1: 'desert',
    2: 'green_area',
    3: 'water'
}

cnn_model.save('model_2025_satellite.keras')

# Conversion

model = models.load_model('model_2025_satellite.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with tf.io.gfile.GFile('model_2025_satellite.tflite', 'wb') as f:
    f.write(tflite_model)