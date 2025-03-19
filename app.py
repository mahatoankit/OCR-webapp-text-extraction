import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import re
import pytesseract
from database import CitizenshipDatabase
import datetime

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Target aspect ratio for document (1.41:1 is close to A4)
TARGET_ASPECT_RATIO = 1.41

db = CitizenshipDatabase()


def check_and_correct_orientation(img):
    """
    Check if the image is in the correct orientation and rotate if needed.
    We want width > height for citizenship card (landscape orientation).
    """
    height, width = img.shape[:2]
    current_ratio = width / height

    # Use Tesseract to detect orientation
    try:
        osd = pytesseract.image_to_osd(img)
        angle = int(osd.split("Rotate: ")[1].split("\n")[0])
        if angle != 0:
            # Rotate the image to correct the orientation
            if angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Update dimensions
            height, width = img.shape[:2]
            current_ratio = width / height
    except Exception as e:
        st.warning("Could not automatically detect document orientation")

    # If image is in portrait mode but should be landscape (width < height)
    if width < height:
        # Rotate 90 degrees
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # Update dimensions
        height, width = img.shape[:2]
        current_ratio = width / height

    return img, current_ratio


def adjust_aspect_ratio(img, target_ratio=TARGET_ASPECT_RATIO):
    """
    Adjust the image to have the target aspect ratio (1.41:1)
    """
    height, width = img.shape[:2]
    current_ratio = width / height

    # If current ratio is close enough to target, return original
    if abs(current_ratio - target_ratio) < 0.1:
        return img

    # Calculate new dimensions to achieve target ratio
    if current_ratio > target_ratio:
        # Too wide, adjust height
        new_height = int(width / target_ratio)
        new_width = width

        # Create a new image with black padding
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Place the original image in the center
        y_offset = (new_height - height) // 2
        new_img[y_offset : y_offset + height, 0:width] = img

    else:
        # Too tall, adjust width
        new_width = int(height * target_ratio)
        new_height = height

        # Create a new image with black padding
        new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Place the original image in the center
        x_offset = (new_width - width) // 2
        new_img[0:height, x_offset : x_offset + width] = img

    return new_img


def preprocess_image(img):
    """Simple preprocessing: convert to grayscale and enhance contrast"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to color for consistency
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# 1. First, reduce the padding value
def calculate_field_positions(width, height):
    # Reduce padding for field boxes (from 0.02 to 0.01)
    padding = 0.01

    # Complete definition for all fields on the front side (removing Issue Date)
    return {
        "नाम थर / Full Name": {
            "label_pos": (int(width * 0.1), int(height * 0.15)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.13 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.20 + padding))),
        },
        "बुवाको नाम / Father's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.25)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.23 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.30 + padding))),
        },
        "आमाको नाम / Mother's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.35)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.33 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.40 + padding))),
        },
        "जन्म मिति / Date of Birth": {
            "label_pos": (int(width * 0.1), int(height * 0.45)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.43 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.50 + padding))),
        },
        "जन्म स्थान / Place of Birth": {
            "label_pos": (int(width * 0.1), int(height * 0.55)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.53 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.60 + padding))),
        },
        "लिङ्ग / Gender": {
            "label_pos": (int(width * 0.1), int(height * 0.65)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.63 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.70 + padding))),
        },
        "नागरिकता नं / Citizenship Number": {
            "label_pos": (int(width * 0.1), int(height * 0.75)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.73 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.80 + padding))),
        },
        "स्थायी ठेगाना / Permanent Address": {
            "label_pos": (int(width * 0.1), int(height * 0.85)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.83 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.90 + padding))),
        },
    }


# 2. Also update the back field positions
def calculate_back_field_positions(width, height):
    # Reduce padding for field boxes (from 0.02 to 0.01)
    padding = 0.01

    # Updated back fields without fingerprint
    return {
        "पति/पत्नीको नाम / Spouse's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.15)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.13 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.25 + padding))),
        },
        "जारी मिति / Issue Date": {
            "label_pos": (int(width * 0.1), int(height * 0.35)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.33 - padding)),
            ),
            "box_end": (int(width * (0.6 + padding)), int(height * (0.40 + padding))),
        },
        "जारी गर्ने अधिकारी / Issuing Authority": {
            "label_pos": (int(width * 0.1), int(height * 0.55)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.53 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.65 + padding))),
        },
    }


def detect_and_label_citizenship_document(image, side="front"):
    # Convert PIL Image to CV2 format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = preprocess_image(img)

    # Check if image is mostly blank/empty
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.mean(gray)[0] > 240:  # If image is mostly white
        st.warning(f"The {side} image appears to be blank or empty.")
        return image, False, None

    # Check if image has enough text content
    text = pytesseract.image_to_string(img)
    if len(text.strip()) < 10:  # Less than 10 chars of text
        st.warning(f"The {side} image doesn't appear to contain enough text content.")
        return image, False, None

    # Continue with existing processing...
    # ...

    # Save the preprocessed dimensions for later reference
    preprocessed_height, preprocessed_width = img.shape[:2]
    aspect_ratio = preprocessed_width / preprocessed_height

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase kernel size for better noise reduction
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adjust Canny parameters
    edges = cv2.Canny(blurred, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)

        # Create result image with green boundary
        result = img.copy()
        cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)

        # Get dimensions
        height, width = result.shape[:2]

        # Define fields based on document side
        front_fields = calculate_field_positions(width, height)
        back_fields = calculate_back_field_positions(width, height)

        fields = front_fields if side == "front" else back_fields

        # Verify field content before labeling
        valid_fields = {}
        for field, positions in fields.items():
            x1, y1 = positions["box_start"]
            x2, y2 = positions["box_end"]

            # Extract just this field area
            field_img = gray[y1:y2, x1:x2]

            # Check if field area has text-like content
            edges = cv2.Canny(field_img, 30, 200)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Only include fields with sufficient content
            if len(contours) > 5:  # A reasonable number of contours indicates text
                valid_fields[field] = positions

        # Only label fields that passed the content check
        fields = valid_fields

        # Draw labels and boxes with unicode font support
        for field, positions in fields.items():
            img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)

            # Use a TrueType font that supports Devanagari
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/unifont/unifont.ttf", 32
                )
            except:
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                        32,
                    )
                except:
                    font = ImageFont.load_default()

            draw.text(positions["label_pos"], field, font=font, fill=(255, 0, 0))
            result = np.array(img_pil)

            # Draw box for the field area with higher visibility
            cv2.rectangle(
                result,
                positions["box_start"],
                positions["box_end"],
                (0, 0, 255),
                2,
            )

            # Add field name inside or near the box for clarity
            img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)
            label_inside_pos = (
                positions["box_start"][0] + 10,
                positions["box_start"][1] + 5,
            )
            text_w, text_h = draw.textbbox((0, 0), field, font=font)[2:]
            draw.rectangle(
                [
                    label_inside_pos[0] - 2,
                    label_inside_pos[1] - 2,
                    label_inside_pos[0] + text_w + 2,
                    label_inside_pos[1] + text_h + 2,
                ],
                fill=(255, 255, 255, 180),
            )
            draw.text(label_inside_pos, field, font=font, fill=(0, 0, 0))
            result = np.array(img_pil)

        return (
            Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)),
            True,
            aspect_ratio,
        )

    return image, False, None


def highlight_extracted_fields(front_image, back_image):
    """
    Function to create a visualization of exactly where data is being extracted.
    This provides a clearer highlight of the extraction areas.
    """
    front_img = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)
    back_img = cv2.cvtColor(np.array(back_image), cv2.COLOR_RGB2BGR)

    front_img = preprocess_image(front_img)
    back_img = preprocess_image(back_img)

    front_height, front_width = front_img.shape[:2]
    back_height, back_width = back_img.shape[:2]

    front_overlay = front_img.copy()
    back_overlay = back_img.copy()

    front_fields = calculate_field_positions(front_width, front_height)
    back_fields = calculate_back_field_positions(back_width, back_height)

    for fields, img in [(front_fields, front_overlay), (back_fields, back_overlay)]:
        for field, positions in fields.items():
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                positions["box_start"],
                positions["box_end"],
                (0, 255, 0),
                -1,
            )
            # Reduce alpha to make highlight more transparent (from 0.4 to 0.3)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # Make border thinner (from 2 to 1)
            cv2.rectangle(
                img,
                positions["box_start"],
                positions["box_end"],
                (0, 0, 255),
                1,
            )

    highlighted_front = Image.fromarray(cv2.cvtColor(front_overlay, cv2.COLOR_BGR2RGB))
    highlighted_back = Image.fromarray(cv2.cvtColor(back_overlay, cv2.COLOR_BGR2RGB))

    return highlighted_front, highlighted_back


def format_extracted_text(text):
    """Format the extracted text into English section with clean output"""
    # Extract all fields first
    extracted = {
        "full_name": extract_field(text, r"(?:नाम थर|Full Name)[:\s]*(.*?)(?:\n|$)"),
        "father_name": extract_field(
            text, r"(?:बुवाको नाम|Father's Name)[:\s]*(.*?)(?:\n|$)"
        ),
        "mother_name": extract_field(
            text, r"(?:आमाको नाम|Mother's Name)[:\s]*(.*?)(?:\n|$)"
        ),
        "dob": extract_field(text, r"(?:जन्म मिति|Date of Birth)[:\s]*(.*?)(?:\n|$)"),
        "birth_place": extract_field(
            text, r"(?:जन्म स्थान|Place of Birth)[:\s]*(.*?)(?:\n|$)"
        ),
        "gender": extract_field(text, r"(?:लिङ्ग|Gender)[:\s]*(.*?)(?:\n|$)"),
        "citizenship_no": extract_field(
            text, r"(?:नागरिकता नं|Citizenship Number)[:\s]*(.*?)(?:\n|$)"
        ),
        "address": extract_field(
            text, r"(?:स्थायी ठेगाना|Permanent Address)[:\s]*(.*?)(?:\n|$)"
        ),
        "spouse_name": extract_field(
            text, r"(?:पति/पत्नीको नाम|Spouse's Name)[:\s]*(.*?)(?:\n|$)"
        ),
        "issue_date": extract_field(
            text, r"(?:जारी मिति|Issue Date)[:\s]*(.*?)(?:\n|$)"
        ),
        "authority": extract_field(
            text, r"(?:जारी गर्ने अधिकारी|Issuing Authority)[:\s]*(.*?)(?:\n|$)"
        ),
    }

    # Remove any duplicate label in Citizenship Number if present
    extracted["citizenship_no"] = re.sub(
        r"^(?:नागरिकता नं\s*[:\-]\s*)", "", extracted["citizenship_no"]
    )

    # Helper function to detect if text contains Nepali characters
    def contains_nepali(text):
        nepali_unicode_range = range(0x0900, 0x097F)  # Devanagari Unicode range
        return any(ord(char) in nepali_unicode_range for char in text)

    # Process fields to extract English content and note where translation is needed
    english_data = {}

    for field, value in extracted.items():
        if value == "Not found":
            english_data[field] = "Not found"
            continue

        # If the field contains Nepali text
        if contains_nepali(value):
            # Extract any English text that might be present
            english_parts = re.sub(r"[\u0900-\u097F]+", "", value).strip()

            if english_parts:
                # If we have some English text, use that
                english_data[field] = english_parts
            else:
                # If only Nepali text, indicate translation needed
                english_data[field] = "[Nepali text needs translation]"
        else:
            # Field is already in English
            english_data[field] = value

    # Create a clean template without markdown for display and download
    template = """
Citizenship Certificate Information

Personal Details
Full Name: {full_name}
Father's Name: {father_name}
Mother's Name: {mother_name}
Gender: {gender}

Document Details
Citizenship Number: {citizenship_no}
Permanent Address: {address}
Date of Birth: {dob}
Place of Birth: {birth_place}
Issue Date: {issue_date}
Issuing Authority: {authority}
Spouse's Name: {spouse_name}
"""

    # Return formatted text with English data
    return template.format(**english_data)


def extract_field(text, pattern):
    """Extract field using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Not found"


# Update Streamlit interface
tab1, tab2 = st.tabs(["Document Scanner", "View Records"])

with tab1:
    # Existing scanner interface
    st.title("Nepali Citizenship Document Scanner")
    st.write("Upload both sides of your citizenship document")

    col1, col2 = st.columns(2)
    with col1:
        front_file = st.file_uploader(
            "Front side", type=["jpg", "jpeg", "png"], key="front"
        )
    with col2:
        back_file = st.file_uploader(
            "Back side", type=["jpg", "jpeg", "png"], key="back"
        )

    # Keep all your existing document processing code here
    if front_file and back_file:
        front_image = Image.open(front_file)
        back_image = Image.open(back_file)

        st.subheader("Uploaded Documents")
        col1, col2 = st.columns(2)
        with col1:
            st.image(front_image, caption="Front Side", use_container_width=True)
        with col2:
            st.image(back_image, caption="Back Side", use_container_width=True)

        if st.button("Extract Information"):
            with st.spinner("Processing images..."):
                # Display labeled images
                labeled_front, flag_front, aspect_front = (
                    detect_and_label_citizenship_document(front_image, side="front")
                )
                labeled_back, flag_back, aspect_back = (
                    detect_and_label_citizenship_document(back_image, side="back")
                )

                st.subheader("Labeled Documents")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(
                        labeled_front,
                        caption="Labeled Front Side",
                        use_container_width=True,
                    )
                with col2:
                    st.image(
                        labeled_back,
                        caption="Labeled Back Side",
                        use_container_width=True,
                    )

                # Preprocess images for text extraction
                front_cv = cv2.cvtColor(np.array(front_image), cv2.COLOR_RGB2BGR)
                back_cv = cv2.cvtColor(np.array(back_image), cv2.COLOR_RGB2BGR)

                front_processed = Image.fromarray(
                    cv2.cvtColor(preprocess_image(front_cv), cv2.COLOR_BGR2RGB)
                )
                back_processed = Image.fromarray(
                    cv2.cvtColor(preprocess_image(back_cv), cv2.COLOR_BGR2RGB)
                )

                # Create prompt for Gemini with processed images
                prompt = [
                    "Extract the following information from this Nepali citizenship document:",
                    "- Full Name (नाम थर)",
                    "- Father's Name (बुवाको नाम)",
                    "- Mother's Name (आमाको नाम)",
                    "- Date of Birth (जन्म मिति)",
                    "- Place of Birth (जन्म स्थान)",
                    "- Gender (लिङ्ग)",
                    "- Citizenship Number (नागरिकता नं)",
                    "- Permanent Address (स्थायी ठेगाना)",
                    "- Spouse's Name (पति/पत्नीको नाम)",
                    "- Issue Date (जारी मिति)",
                    "- Issuing Authority (जारी गर्ने अधिकारी)",
                    "Provide both Nepali and English text when available.",
                    front_processed,
                    back_processed,
                ]

                try:
                    response = model.generate_content(prompt)

                    if response and response.text:
                        st.subheader("Extracted Information")
                        formatted_text = format_extracted_text(response.text)

                        # Display text as before
                        display_text = formatted_text.replace(
                            "Personal Details", "### Personal Details"
                        )
                        display_text = display_text.replace(
                            "Document Details", "### Document Details"
                        )
                        display_text = re.sub(
                            r"^(.*?): ", r"- **\1:** ", display_text, flags=re.MULTILINE
                        )
                        st.markdown(display_text)

                        # Parse data for database
                        data_lines = formatted_text.strip().split("\n")
                        data_dict = {}

                        for line in data_lines:
                            if ": " in line:
                                field, value = line.split(": ", 1)
                                field = field.strip().lower().replace(" ", "_")
                                # Handle the field name difference
                                if field == "permanent_address":
                                    field = "address"
                                data_dict[field] = value.strip()

                        # Add Save and Download buttons side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save to Database"):
                                success, message = db.save_record(data_dict)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)

                        with col2:
                            st.download_button(
                                "Download as Text",
                                formatted_text,
                                file_name="citizenship_data.txt",
                            )
                    else:
                        st.error("No information could be extracted")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    # Records view tab
    st.title("Saved Citizenship Records")

    records = db.get_all_records()

    if records:
        for record in records:
            record_id = record[0]
            name = record[1]
            citizenship_no = record[2]
            scan_date = record[3]

            with st.expander(f"{name} - {citizenship_no}"):
                if st.button("View Details", key=f"view_{record_id}"):
                    record_details = db.get_record_by_id(record_id)
                    if record_details:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("### Personal Details")
                            st.write(f"**Full Name:** {record_details[1]}")
                            st.write(f"**Father's Name:** {record_details[2]}")
                            st.write(f"**Mother's Name:** {record_details[3]}")
                            st.write(f"**Gender:** {record_details[4]}")
                            st.write(f"**Spouse Name:** {record_details[11]}")

                        with col2:
                            st.write("### Document Details")
                            st.write(f"**Citizenship No:** {record_details[5]}")
                            st.write(f"**Address:** {record_details[6]}")
                            st.write(f"**Date of Birth:** {record_details[7]}")
                            st.write(f"**Place of Birth:** {record_details[8]}")
                            st.write(f"**Issue Date:** {record_details[9]}")
                            st.write(f"**Authority:** {record_details[10]}")
                            st.write(f"**Scan Date:** {record_details[12]}")
    else:
        st.info("No citizenship records found in the database")