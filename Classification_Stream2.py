from transformers import pipeline
import streamlit as st
from PIL import Image

# Load the image-to-text pipeline with the selected Hugging Face model
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Streamlit app setup
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate the caption
    st.write("Generating caption...")
    caption = image_to_text(image)[0]["generated_text"]
    
    # Display the generated caption
    st.write("Caption:")
    st.write(caption)
