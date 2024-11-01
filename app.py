import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Constants
IMAGE_SIZE = 256
CHANNELS = 3
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load the trained model with caching
@st.cache_resource
def load_trained_model():
    model = load_model("models/version_1.keras")  # Adjust the path if needed
    return model

model = load_trained_model()

# Preprocessing function
def preprocess_image(uploaded_img):
    img = Image.open(uploaded_img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize image to model input dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Scale the image
    return img_array

# Prediction function
def predict(model, preprocessed_img):
    predictions = model.predict(preprocessed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit app layout
st.title("Potato Disease Detection")
st.write("Upload a potato leaf image to check for Early Blight, Late Blight, or a Healthy leaf.")

uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    preprocessed_img = preprocess_image(uploaded_file)
    
    # Display the uploaded image
    st.image(preprocessed_img[0], channels="RGB", caption="Uploaded Image", use_column_width=True)
    
    # Get prediction
    predicted_class, confidence = predict(model, preprocessed_img)
    
    # Display prediction
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence}%")
