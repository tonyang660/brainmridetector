import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from brain_tumor_classifier import load_model, predict_image

# Title of the app
st.title('Brain MRI Tumor Detector 🧠')
st.write('💊 Healthcare Sector - By Thomas, Sam, Ioannis, Aian and Tony - AI Accelerator Program Fall 2024')
st.divider()

# File uploader for image input
image_input = st.file_uploader("Please upload a Brain MRI image: ", type=["jpg", "png", "jpeg"])

if image_input is not None:
    # Load and preprocess the uploaded image
    image = Image.open(image_input)
    resized_image = image.resize((224, 224)).convert('RGB')  # Resize and convert to RGB
    st.image(resized_image, caption="Uploaded image")

    # Convert image to NumPy array and normalize
    image_array = np.array(resized_image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Load the trained model
    model = load_model('MRI_Anomaly_Detection_Model.keras')  # Ensure this path is correct

    # Make a prediction
    prediction = model.predict(image_array)

    # Map the predicted index to class names
    class_names = ['other', 'glioma', 'meningioma', 'notumor', 'pituitary']  # Replace with actual class names if different
    predicted_class = class_names[np.argmax(prediction)]

    # Display the prediction result
    st.success(f"The model predicts: **{predicted_class}**")
    