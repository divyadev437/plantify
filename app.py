import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time

# -----------------------------
# Load Model (with Caching)
# -----------------------------
# This function will only run once, making the app much faster.
@st.cache_resource
def load_app_model():
    """Loads and caches the Keras model."""
    try:
        model = tf.keras.models.load_model("trained_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure 'trained_model.keras' is the correct binary model file and is in the same folder as app.py.")
        return None

model = load_app_model()

# -----------------------------
# Load Class Names (with Caching)
# -----------------------------
@st.cache_data
def load_class_names():
    """Loads and caches the class names from JSON."""
    try:
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        st.error("Error: 'class_names.json' not found.")
        return []

class_names = load_class_names()

# -----------------------------
# Load Disease Info (with Caching)
# -----------------------------
@st.cache_data
def load_disease_info():
    """Loads and caches the disease information from JSON."""
    try:
        with open("disease_info.json", "r") as f:
            disease_info = json.load(f)
        return disease_info
    except FileNotFoundError:
        st.error("Error: 'disease_info.json' not found.")
        return {}

disease_info = load_disease_info()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Plantify Disease Detection", page_icon="🌿", layout="wide")

# --- Attractive Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌿 Plantify Disease Detection 🌿</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Upload a leaf image and let AI analyze its health.</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "This app uses AI to detect plant diseases from leaf images. "
    "Upload an image or use your camera to get started!"
)
st.sidebar.title("📸 Upload Options")
upload_option = st.sidebar.radio("Choose input method:", ("Upload Image", "Camera Capture"))

# --- Main Layout (2 Columns) ---
col1, col2 = st.columns([1, 1.2]) # Give more space to the results column
image = None

with col1:
    st.header("1. Provide an Image")
    
    if upload_option == "Upload Image":
        # Expanded file types
        uploaded_file = st.file_uploader("Click to upload a plant leaf image:", type=["jpg", "png", "jpeg", "bmp", "webp"])
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error opening image file: {e}. Please try a different file.")
                image = None

    elif upload_option == "Camera Capture":
        camera_input = st.camera_input("Take a picture of the leaf")
        if camera_input:
            image = Image.open(camera_input)
    
    if image:
        st.image(image, caption="Your Image", use_column_width=True)

with col2:
    st.header("2. Analysis Result")
    
    if image and model and class_names and disease_info:
        # --- Prediction Logic ---
        with st.spinner("🧠 Analyzing the image... Please wait."):
            time.sleep(1) # Small delay to make spinner visible
            
            # Preprocess
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0) # (1, 128, 128, 3)

            # Predict
            predictions = model.predict(img_array)
            predicted_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class = class_names[predicted_idx]

        # --- Attractive Result Display ---
        st.subheader("✅ Prediction")
        
        # Use columns for a cleaner metric display
        c1, c2 = st.columns(2)
        c1.metric("Detected Disease:", predicted_class)
        c2.metric("Confidence:", f"{confidence*100:.2f}%")

        if confidence > 0.85:
            st.success("High confidence in this prediction.")
        elif confidence > 0.6:
            st.warning("Moderate confidence. The result may be less accurate.")
        else:
            st.error("Low confidence. Please try a clearer, more centered image.")

        # -----------------------------
        # Disease Information with Tabs
        # -----------------------------
        st.markdown("---")
        st.subheader("📝 Disease Information")

        if predicted_class in disease_info:
            info = disease_info[predicted_class]
            
            # Handle the "healthy" case separately
            if "healthy" in predicted_class.lower():
                st.balloons()
                st.success(f"**Status: {info.get('title', 'Healthy')}** 🎉")
                st.write(info.get("description", "The plant appears to be in good health. Keep up the good care!"))
                st.write(f"**Symptoms:** {info.get('symptoms', 'N/A')}")
                st.write(f"**Prevention:** {info.get('prevention', 'N/A')}")
            
            else:
                # Use tabs for a clean look
                tab_titles = ["Description", "Symptoms", "Prevention", "Remedy"]
                tab1, tab2, tab3, tab4 = st.tabs(tab_titles)
                
                with tab1:
                    st.subheader(info.get("title", predicted_class))
                    st.write(info.get("description", "No description available."))
                
                with tab2:
                    st.write(info.get("symptoms", "No symptoms information available."))
                
                with tab3:
                    st.write(info.get("prevention", "No prevention information available."))
                
                with tab4:
                    st.write(info.get("remedy", "No remedy information available."))
                    
        else:
            st.error("Information for this disease is not available in the database.")
    
    elif not (model and class_names and disease_info):
        st.error("App is not fully loaded. Check for file loading errors above.")
    else:
        st.info("Please upload an image or use the camera to get an analysis.")