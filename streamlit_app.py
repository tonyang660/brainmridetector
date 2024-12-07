import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from main import load_model, predict_image

# Title of the app
st.title('Brain MRI Tumor Detector 🧠')
st.write('💊 Healthcare Sector - By Thomas, Sam, Ioannis, Aian and Tony - AI Accelerator Program Fall 2024')
st.divider()

# File uploader for image input
image_input = st.file_uploader("Please upload a Brain MRI image: ", type=["jpg", "png", "jpeg"])

if image_input is not None:
    image = Image.open(image_input)
    
    # Resize image and ensure it's in RGB mode
    resized_image = image.resize((224, 224)).convert('RGB')
    st.image(resized_image, caption="Uploaded image")

