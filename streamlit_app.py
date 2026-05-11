# streamlit_app.py

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("🐱🐶 Cat vs Dog Classifier")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_dogs_classifier.h5")


model = load_model()

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
            img = image.convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            result = "Dog" if prediction[0][0] > 0.5 else "Cat"

            st.success(f"Prediction: {result}")

        except Exception as e:
            st.error(f"Error: {e}")