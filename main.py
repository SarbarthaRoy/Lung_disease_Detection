import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Set the image file path for prediction
image_path ='/content/drive/MyDrive/Disease/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset/covid/COVID-19 (102).jpg'

# Load the saved model
model = load_model('/content/Model.h5')

# Load and preprocess the input image
img = load_img(image_path, target_size=image_shape)
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image data to [0, 1]

# Make predictions
predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_class = np.argmax(predictions[0])

# Map the predicted class index to its corresponding label
class_labels = {0: 'normal', 1: 'pneumonia', 2: 'covid'}
predicted_label = class_labels[predicted_class]

print("Predicted Disease:", predicted_label)