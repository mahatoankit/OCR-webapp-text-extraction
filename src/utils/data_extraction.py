import os
import google.generativeai as genai
from PIL import Image
import io
import streamlit as st
import json
from datetime import datetime

# Configure Gemini
genai.configure(api_key="AIzaSyCotnndBVRVBcGlHUiKbP_I_mTmAr1Ages")
model = genai.GenerativeModel("gemini-2.0-flash")


def extract_citizenship_data(image_bytes, is_front=True):
    """
    Extract citizenship data using Gemini model

    Args:
        image_bytes: Image bytes to process
        is_front: Boolean indicating if it's front side

    Returns:
        str: JSON string with extracted data
    """
    if is_front:
        prompt = """
        Analyze this front side of Nepali citizenship document carefully and extract these specific fields in JSON format:
        {
            "full_name": "complete name in English",
            "father_name": "father's name in English",
            "mother_name": "mother's name in English",
            "gender": "Male or Female",
            "citizenship_no": "the citizenship number exactly as it appears",
            "dob": "date of birth in YYYY-MM-DD format if possible",
            "birth_place": "place of birth in English",
            "issue_date": "issue date in YYYY-MM-DD format if possible, issue date can be in nepali check properly",
            "authority": "name of the issuing authority/office"
        }

        Look carefully for Nepali text (देवनागरी) and translate to English when needed.
        Focus only on extracting these fields - no additional information.
        Return clean JSON with no markdown formatting or explanations.
        """
    else:
        prompt = """
        Analyze this back side of Nepali citizenship document carefully and extract these specific fields in JSON format:
        {
            "permanent_address": "complete permanent address in English",
            "spouse_name": "spouse's name in English if present"
        }

        Look carefully for Nepali text (देवनागरी) and translate to English when needed.
        Focus only on extracting these fields - no additional information.
        Return clean JSON with no markdown formatting or explanations.
        """

    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Pass image to the model
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None


def extract_data_from_images(front_image_bytes, back_image_bytes):
    """Extract data from both front and back images of the citizenship document"""
    front_result = extract_citizenship_data(front_image_bytes, is_front=True)
    back_result = extract_citizenship_data(back_image_bytes, is_front=False)

    # Clean up and parse JSON responses
    front_data = None
    back_data = None

    if front_result:
        try:
            front_json_str = (
                front_result.strip().replace("```json", "").replace("```", "")
            )
            front_data = json.loads(front_json_str)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing front image data: {str(e)}")

    if back_result:
        try:
            back_json_str = (
                back_result.strip().replace("```json", "").replace("```", "")
            )
            back_data = json.loads(back_json_str)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing back image data: {str(e)}")

    # Merge data and add scan date
    combined_data = {}

    if front_data:
        combined_data.update(front_data)

    if back_data:
        combined_data.update(back_data)

    # Add scan date
    combined_data["scan_date"] = datetime.now().strftime("%Y-%m-%d")

    return combined_data
