import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Clothing Classification",
    page_icon="👕",
    layout="centered"
)

# Clothing categories
# Must match the training class mapping
class_names = [
    "Blazer",
    "Blouse",
    "Body",
    "Dress",
    "Hat",
    "Hoodie",
    "Longsleeve",
    "Outwear",
    "Pants",
    "Polo",
    "Shirt",
    "Shoes",
    "Shorts",
    "Skirt",
    "T-Shirt",
    "Top",
    "Undershirt"
]

# Load trained model
@st.cache_resource
def load_trained_model():
    model = load_model(
        "resnet50_finetuned.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
        safe_mode=False
    )
    return model

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title
st.title("👕 Clothing Classification")
st.write(
    "Classify clothing images using a "
    "Fine-Tuned ResNet50 Deep Learning Model."
)

st.info("Model Test Accuracy: 82.61%")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Clothing Image",
        use_container_width=True
    )

    # Prediction
    if st.button("Predict Clothing Category"):

        # Resize image
        img = image.resize((224, 224))

        # Convert to NumPy array
        img_array = np.array(
            img,
            dtype=np.float32
        )

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Model already includes ResNet50
        # preprocessing in its Lambda layer.
        # Do not preprocess the image again.

        # Predict
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            np.max(predictions[0])
        ) * 100

        # Display results
        st.success(
            f"Predicted Clothing Category: "
            f"{predicted_class}"
        )

        st.write(
            f"Prediction Confidence: "
            f"{confidence:.2f}%"
        )

        st.progress(
            min(int(confidence), 100)
        )

        # Display top 3 predictions
        st.subheader("Top 3 Predictions")

        top_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]

        for idx in top_indices:
            st.write(
                f"{class_names[idx]}: "
                f"{predictions[0][idx] * 100:.2f}%"
            )

st.divider()

st.caption(
    "CNN Capstone Project 2 | "
    "Fine-Tuned ResNet50 | "
    "17 Clothing Categories"
)