import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageFilter

# Load model and PCA
model = joblib.load("logreg_mnist_model.pkl")
pca = joblib.load("mnist_pca_transform.pkl")

# Page config
st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("🧠 MNIST Digit Classifier using Logistic Regression + PCA")

st.markdown("""
Upload a handwritten digit image (preferably 28x28 or square shape), and the model will predict the digit.
""")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Invert if background is white
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = Image.fromarray(255 - img_np)

    # Apply blur using PIL
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    img_np = np.array(img) / 255.0
    img_flat = img_np.reshape(1, -1)
    return img_flat, img_np

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing and predicting..."):
        img_flat, img_resized = preprocess_image(image)
        img_pca = pca.transform(img_flat)
        prediction = model.predict(img_pca)

    st.success(f"✅ Predicted Digit: **{prediction[0]}**")

    st.subheader("Processed Input")
    st.image(img_resized, width=200, caption="Preprocessed 28x28 Input", clamp=True, channels="GRAY")
