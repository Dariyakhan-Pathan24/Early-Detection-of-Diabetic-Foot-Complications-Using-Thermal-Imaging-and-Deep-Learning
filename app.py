# app.py
# ------------------------------------------------------
# Streamlit App with CORRECT label mapping
# ------------------------------------------------------

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
MODEL_PATH = "saved_models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.set_page_config(page_title="Thermal Image Classifier")

st.title("🔥 Diabetic Thermal Image Analysis")
st.write("Upload a thermal image to classify as Normal or Abnormal.")

# ⚠️ IMPORTANT: Match class order from training
# TensorFlow sorts folder names alphabetically
# So likely: ['abnormal', 'normal']
CLASS_NAMES = ['abnormal', 'normal']

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (same as training)
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prob = model.predict(img_array)[0][0]

    st.write(f"🔍 Raw Model Output (Sigmoid): {prob:.4f}")

    # ✅ Correct label mapping
    if prob > 0.5:
        predicted_class = CLASS_NAMES[1]  # index 1
        confidence = prob
    else:
        predicted_class = CLASS_NAMES[0]  # index 0
        confidence = 1 - prob

    # Convert to display label
    if predicted_class == "normal":
        display_label = "✅ Normal"
    else:
        display_label = "⚠️ Abnormal"

    # Show result
    st.subheader(f"Prediction: {display_label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    # Confidence warning
    if confidence < 0.7:
        st.warning("⚠️ Low confidence prediction. Please verify.")

    # Uncertainty zone
    if 0.4 < prob < 0.6:
        st.warning("⚠️ Model is uncertain. This is a borderline case.")

    # Info section
    st.info(
        "This model uses a custom CNN trained on thermal images. "
        "High temperature regions (red/yellow) may indicate conditions."
    )