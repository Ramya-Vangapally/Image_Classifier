# streamlit_app.py

import streamlit as st
from PIL import Image
from app import predict_image

st.title("🐱🐶 Cat vs Dog Classifier")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Predict button
    if st.button("Predict"):
        try:
            # Convert to RGB before saving to prevent RGBA to JPEG error for PNGs
            rgb_image = image.convert('RGB')
            # Save temporarily
            file_path = "temp.jpg"
            rgb_image.save(file_path)
            
            # Use the predict function from Flask app
            result = predict_image(file_path)
            st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error: {e}")