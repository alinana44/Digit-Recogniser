# app.py
import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🖍️ Handwritten Digit Recognition (MNIST + Logistic Regression)")
st.write("Upload a 28x28 grayscale image of a digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert if needed
    img = img.resize((28, 28))
    
    st.image(img, caption="Uploaded Image", use_column_width=False)

    # Preprocess
    img_array = np.array(img).reshape(1, -1)
    img_scaled = scaler.transform(img_array)

    # Predict
    prediction = model.predict(img_scaled)
    st.success(f"Predicted Digit: {prediction[0]}")
