# End-to-End Deep Learning: Cats vs. Dogs Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imageclassifier-model.streamlit.app/)

## 📖 Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [Live Demo](#live-demo)
3. [Dataset & Preparation](#dataset--preparation)
4. [Phase 1: Model Building and Architecture](#phase-1-model-building-and-architecture)
5. [Phase 2: Inference and Deployment](#phase-2-inference-and-deployment)
6. [Project Structure](#project-structure)
7. [Installation & Local Setup](#installation--local-setup)
8. [Usage Instructions](#usage-instructions)
9. [Future Enhancements](#future-enhancements)

---

## 🧐 Project Overview & Problem Statement

Distinguishing between images of cats and dogs is a foundational computer vision problem. While humans do this effortlessly, teaching a machine requires recognizing complex, non-linear pixel patterns.

The goal of this project is to develop an **end-to-end AI-powered Pet Classification System**. Rather than stopping at a Jupyter Notebook experiment, this project covers the entire pipeline: from data extraction and preprocessing, to training a Convolutional Neural Network (CNN) using Transfer Learning, and finally deploying the trained model to the cloud using Streamlit. This proves that deep learning models can be actively shipped to production.

---

## 🚀 Live Demo
Access the fully functional, globally deployed classification model here:  
👉 **[https://imageclassifier-model.streamlit.app/](https://imageclassifier-model.streamlit.app/)**

---

## 💾 Dataset & Preparation

The raw data comes from the well-known **Kaggle Cats vs. Dogs dataset**, which originally contains ~25,000 images. To demonstrate a streamlined and computationally efficient pipeline, we enforce strict dataset constraints:
- **Cat Images Used:** 1,000
- **Dog Images Used:** 1,000
- **Total Images:** 2,000

### Data Preprocessing Steps:
1. **Cleaning:** Iterating through the images using `PIL.Image` and verifying integrity. Corrupted images are automatically bypassed.
2. **Resizing:** Standardizing all inputs to exactly `224x224` pixels to fit the expected temporal shapes of the chosen CNN.
3. **Normalization:** Pixel scaling by factor `/255.0` to force gradient descent conversions into the optimized runtime range of `[0, 1]`.
4. **Data Splitting:** Training vs. validation partitioning ensures the model is monitored for overfitting during the training stage.

---

## 🧠 Phase 1: Model Building and Architecture

Training CNNs from scratch on small datasets generally yields poor accuracy. To solve this, **Transfer Learning** is utilized. 

### Base Model: MobileNetV2
- Pre-trained on **ImageNet** (1.4 million images, 1000 classes).
- **Why MobileNetV2?** It uses Depthwise Separable Convolutions, drastically reducing the number of trainable parameters while maintaining high-grade accuracy. Crucially, it's fast, lightweight, and deployment-friendly—ideal for a Streamlit Cloud environment with resource limits.
- `include_top=False`: We strip the native 1000-class predictor head to add our own.

### Custom Classification Head
We bolt the following top-layers onto the frozen base of MobileNetV2:
1. **Global Average Pooling 2D:** Flattens the localized feature maps into a 1D vector. Superior to a standard `Flatten()` as it reduces parameter explosion and naturally mitigates overfitting.
2. **Dense Layer (128 units):** Fully connected layer using a `ReLU` activation function to interpret the complex extracted features.
3. **Dropout (0.3):** Randomly zeroes out 30% of neurons during runtime to force the network to learn robust, generalized patterns rather than memorizing training data.
4. **Output Dense Layer (1 unit):** Uses a `Sigmoid` activation function, transforming the logits into a probability probability distribution spanning `[0.0, 1.0]`.  

### Training Constraints
- Compiled specifically using **Adam Optimizer** and **Binary Cross-entropy** loss.
- Trained quickly over **5 Epochs**, yielding an optimized frozen file: `cats_dogs_classifier.h5`.

---

## 🌐 Phase 2: Inference and Deployment

Building the architecture is only half the battle. Predicting real-world dynamic data effectively required implementing an inference script.

### 1. Image Inference Logic
When a user uploads a new file:
- The backend leverages Keras's `load_img` processing the raw bits back into a `224x224` structural shape.
- `img_to_array` extracts multi-channel RGB data.
- Numpy `expand_dims` inflates the 3D tensor to 4D `(1, 224, 224, 3)`, mimicking a batch size of 1.
- `model.predict()` emits a probabilistic fraction. If `prediction > 0.5`, result is **Dog**, else **Cat**.

### 2. Streamlit Web Wrapper
We avoid heavyweight boilerplate servers like Django or Flask for the prototype. **Streamlit** binds Python scripts immediately into HTML/React web apps. A slick UI widget captures image drops, caches the `.h5` model to prevent redundant I/O reads, and paints a loading state.

### 3. Cloud Availability (CI/CD Concept)
Tied directly into the Github repository, the web interface pushes to **Streamlit Cloud** directly loading environments off the provided `requirements.txt`.

---

## 📂 Project Structure

```text
📦 imageClassifier
 ┣ 📜 app.py                  # Main functional logic / testing endpoints
 ┣ 📜 cats_dogs_classifier.h5 # Trained & Exported Keras/Tensorflow model
 ┣ 📜 requirements.txt        # Deployment & Virtual environment library lists
 ┣ 📜 runtime.txt             # Required Python versioning details
 ┣ 📜 streamlit_app.py        # Streamlit frontend wrapper and UI configuration
 ┗ 📜 README.md               # Extensive project documentation
```

---

## ⚙️ Installation & Local Setup

To replicate this environment natively on your machine, ensure you have Python 3.8+ installed. 

**1. Clone the Repository or Navigate to the Folder**
```bash
cd desktop\imageClassifier
```

**2. Initialize a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/Scripts/activate  # On Windows
# OR
source env/bin/activate      # On Mac/Linux
```

**3. Install Dependencies from the Requirements File**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Instructions

To boot up the interactive localhost web app running the Model inference:
```bash
streamlit run streamlit_app.py
```
This will automatically compile your scripts and open `http://localhost:8501/` in your default web browser. Click "Browse Files" and upload any `.jpg` or `.png` of a cat or dog to see the algorithm in action!

---

## 🔮 Future Enhancements
- Expand predictive bounds directly to multi-class classification (e.g., adding Birds, Rabbits).
- Replace Streamlit with a FastAPI backend & React frontend for larger scalability.
- Optimize the `.h5` model weights using ONNX or TensorFlow Lite to further reduce operational latency.
