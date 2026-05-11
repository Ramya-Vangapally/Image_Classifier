from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("cats_dogs_classifier.h5")


@app.route('/')
def home():
    return '''
    <h1>Cats vs Dogs Classifier</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload and Predict">
    </form>
    '''


@app.route('/predict', methods=['POST'])
def predict():

    # Get uploaded file
    file = request.files['file']

    if not file:
        return jsonify({
            "error": "No file uploaded"
        })

    # Save temporarily
    file_path = "temp.jpg"
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Convert probability to label
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"

    # Return JSON response
    return jsonify({
        "prediction": result
    })


if __name__ == '__main__':
    app.run(debug=True)