import streamlit as st
from inference import predict_breed
from preprocess import preprocess_image
import config

# Configure the page to be centered
st.set_page_config(
    page_title="Dog Breed Identifier 🐶",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Dog Breed Identifier 🐶")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display image with controlled width to avoid scroll
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    image = preprocess_image(uploaded_file)
    
    with st.spinner('Predicting...'):  # Optional nice touch
        
     breed, confidence = predict_breed(image)  # 🛠 Now returns 2 values!
    
    st.success(f"Predicted Breed: {breed} ({confidence*100:.2f}% confidence)")