# Streamlit app code here:
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('/kaggle/working/finetuned_model.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Streamlit app interface
st.title("Image Classification with ResNet-50")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_array = load_and_preprocess_image(uploaded_file)
    
    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display the prediction
    st.write(f"Predicted class: {predicted_class[0]}")

