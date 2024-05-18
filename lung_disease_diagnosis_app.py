import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Set the image size and load the model
image_shape = (128, 128)
model = load_model('Model.h5')  # Make sure the model file is in the same directory or provide the correct path

# Mapping from class index to human-readable label
class_labels = {0: 'normal', 1: 'pneumonia', 2: 'covid'}

def predict_disease(image):
    # Preprocess the image
    img = image.resize(image_shape)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image data to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    return predicted_label

# Streamlit UI
st.title("Lung Disease Diagnosis System")
st.write("Upload a chest X-ray image to diagnose lung disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)
    st.write("")
    st.write("Analyzing...")

    # Predict the disease
    predicted_label = predict_disease(image)
    st.write(f"The predicted disease is: **{predicted_label}**")
