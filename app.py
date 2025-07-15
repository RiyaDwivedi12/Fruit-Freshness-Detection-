<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fruit_classifier_model.h5")

# Configuration
IMAGE_SIZE = (150, 150)
st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")

# Title Section
st.markdown(
    "<h1 style='text-align: center; color: green;'>Fruit Freshness Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload a fruit image (apple, banana, or orange) to check if it's <b>Fresh</b> or <b>Rotten</b>!</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Image Upload Section
uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for display (smaller size)
    display_image = image.resize((200, 200))
    st.image(display_image, caption="Uploaded Fruit Image", use_column_width=False)

    # Preprocess image for prediction
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Prediction
    prediction = model.predict(img_array)[0][0]
    is_fresh = prediction < 0.5
    label = "Fresh" if is_fresh else "Rotten"
    confidence = (1 - prediction) if is_fresh else prediction

    # Result Section
    st.markdown("### 🧠 Prediction Result")
    st.markdown(
        f"<h2 style='color: {'green' if is_fresh else 'red'};'>{label}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("#### 🔍 Confidence Level")
    st.progress(float(confidence))
    st.markdown(
        f"<p style='font-size:18px;'>Confidence Score: <b>{confidence:.2f}</b></p>",
        unsafe_allow_html=True,
    )
else:
    st.info("👆 Please upload an image file to begin.")
=======
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fruit_classifier_model.h5")

# Configuration
IMAGE_SIZE = (150, 150)
st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎")

# Title Section
st.markdown(
    "<h1 style='text-align: center; color: green;'>Fruit Freshness Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Upload a fruit image (apple, banana, or orange) to check if it's <b>Fresh</b> or <b>Rotten</b>!</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Image Upload Section
uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize for display (smaller size)
    display_image = image.resize((200, 200))
    st.image(display_image, caption="Uploaded Fruit Image", use_column_width=False)

    # Preprocess image for prediction
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Prediction
    prediction = model.predict(img_array)[0][0]
    is_fresh = prediction < 0.5
    label = "Fresh" if is_fresh else "Rotten"
    confidence = (1 - prediction) if is_fresh else prediction

    # Result Section
    st.markdown("### 🧠 Prediction Result")
    st.markdown(
        f"<h2 style='color: {'green' if is_fresh else 'red'};'>{label}</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("#### 🔍 Confidence Level")
    st.progress(float(confidence))
    st.markdown(
        f"<p style='font-size:18px;'>Confidence Score: <b>{confidence:.2f}</b></p>",
        unsafe_allow_html=True,
    )
else:
    st.info("👆 Please upload an image file to begin.")
>>>>>>> 2a0e9f45d3c51f1c2f3eba687c79368b6ef132d0
