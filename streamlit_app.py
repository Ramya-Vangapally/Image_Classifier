# streamlit_app.py

import streamlit as st
import requests
from PIL import Image

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
            # Send image to Flask backend
            files = {"file": uploaded_file.getvalue()}

            response = requests.post(
                "http://127.0.0.1:5000/predict",
                files=files
            )

            result = response.json()

            st.success(f"Prediction: {result['prediction']}")

        except Exception as e:
            st.error(f"Error: {e}")