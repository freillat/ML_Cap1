# import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

# url = 'https://plus.unsplash.com/premium_photo-1663952767362-e95f11d98acd?q=80&w=1550&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'

interpreter = tflite.Interpreter(model_path='model_2025_satellite.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
output_index = output_details[0]['index']

labels = {
    0: 'cloudy',
    1: 'desert',
    2: 'green_area',
    3: 'water'
}

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(X):
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return labels[preds[0].argmax()]

def prediction(url):
    image = download_image(url)
    prepped_img = prepare_image(image, (256, 256))
    X = np.array(prepped_img, dtype=np.float32)
    X /= 255.0
    X = np.expand_dims(X, axis=0)
    return predict(X)

def lambda_handler(event, context):
    url = event['url']
    result = prediction(url)
    return result

# print(prediction(url))