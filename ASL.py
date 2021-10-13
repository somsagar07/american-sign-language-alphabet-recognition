# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:22:44 2021

@author: Sagar
"""

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = tf.keras.models.load_model('keras_model.h5', compile=False)

with open('labels.txt', 'r') as f:
    name = f.read().split('\n')

# Create the array of the right shape to feed into the keras model

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open('test/aslb.jpeg')
#resize the image to a 224x224 
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 255.0)

data[0] = normalized_image_array

prediction = model.predict(data)
index = np.argmax(prediction)
name = name[index] 
print(name[2])
