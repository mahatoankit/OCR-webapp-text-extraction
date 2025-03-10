import streamlit as st
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

st.title("Handwritten Text Recognition with Gemini Vision")
st.write("Upload an image to extract handwritten text.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate text from image
    st.write("Extracting text...")
    try:
        response = model.generate_content(["Please extract and transcribe any text from this image:", image])
        extracted_text = response.text
        
        # Display extracted text
        st.write("Extracted Text:")
        st.write(extracted_text)
    except Exception as e:
        st.error(f"Error in text extraction: {str(e)}")