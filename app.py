import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os
from dotenv import load_dotenv
import cv2
import numpy as np
import re
import pytesseract

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Target aspect ratio for document (1.41:1 is close to A4)
TARGET_ASPECT_RATIO = 1.41


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
        st.warning(f"Could not detect orientation using Tesseract: {e}")

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


# 1. Remove Issue Date from front page fields
def calculate_field_positions(width, height):
    # Add some padding to field boxes
    padding = 0.02

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


# 1. Remove fingerprint from back field positions
def calculate_back_field_positions(width, height):
    # Add some padding to field boxes
    padding = 0.02

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

    # Save the preprocessed dimensions for later reference
    preprocessed_height, preprocessed_width = img.shape[:2]
    aspect_ratio = preprocessed_width / preprocessed_height

    # Display aspect ratio information
    st.info(
        f"Preprocessed image dimensions: {preprocessed_width}x{preprocessed_height}, Aspect ratio: {aspect_ratio:.2f}:1"
    )

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
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.rectangle(
                img,
                positions["box_start"],
                positions["box_end"],
                (0, 0, 255),
                2,
            )

    highlighted_front = Image.fromarray(cv2.cvtColor(front_overlay, cv2.COLOR_BGR2RGB))
    highlighted_back = Image.fromarray(cv2.cvtColor(back_overlay, cv2.COLOR_BGR2RGB))

    return highlighted_front, highlighted_back


def format_extracted_text(text):
    """Format the extracted text into a clean structure"""
    template = """
Citizenship Certificate Information

Personal Details:
- Full Name (नाम थर): {full_name}
- Father's Name (बुवाको नाम): {father_name}
- Mother's Name (आमाको नाम): {mother_name}
- Gender (लिङ्ग): {gender}

Document Details:
- Citizenship Number (नागरिकता नं): {citizenship_no}
- Permanent Address (स्थायी ठेगाना): {address}
- Date of Birth (जन्म मिति): {dob}
- Place of Birth (जन्म स्थान): {birth_place}
- Issue Date (जारी मिति): {issue_date}
- Issuing Authority (जारी गर्ने अधिकारी): {authority}
- Spouse's Name (पति/पत्नीको नाम): {spouse_name}
"""
    extracted = {
        "full_name": extract_field(text, r"(?:नाम थर|Full Name)[:\s]*(.*?)(?:\n|$)"),
        "father_name": extract_field(text, r"(?:बुवाको नाम|Father's Name)[:\s]*(.*?)(?:\n|$)"),
        "mother_name": extract_field(text, r"(?:आमाको नाम|Mother's Name)[:\s]*(.*?)(?:\n|$)"),
        "dob": extract_field(text, r"(?:जन्म मिति|Date of Birth)[:\s]*(.*?)(?:\n|$)"),
        "birth_place": extract_field(text, r"(?:जन्म स्थान|Place of Birth)[:\s]*(.*?)(?:\n|$)"),
        "gender": extract_field(text, r"(?:लिङ्ग|Gender)[:\s]*(.*?)(?:\n|$)"),
        "citizenship_no": extract_field(text, r"(?:नागरिकता नं|Citizenship Number)[:\s]*(.*?)(?:\n|$)"),
        "address": extract_field(text, r"(?:स्थायी ठेगाना|Permanent Address)[:\s]*(.*?)(?:\n|$)"),
        "spouse_name": extract_field(text, r"(?:पति/पत्नीको नाम|Spouse's Name)[:\s]*(.*?)(?:\n|$)"),
        "issue_date": extract_field(text, r"(?:जारी मिति|Issue Date)[:\s]*(.*?)(?:\n|$)"),
        "authority": extract_field(text, r"(?:जारी गर्ने अधिकारी|Issuing Authority)[:\s]*(.*?)(?:\n|$)"),
    }
    # Remove any duplicate label in Citizenship Number if present.
    extracted["citizenship_no"] = re.sub(r"^(?:नागरिकता नं\s*[:\-]\s*)", "", extracted["citizenship_no"])

    return template.format(**extracted)


def extract_field(text, pattern):
    """Extract field using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Not found"


# Update Streamlit interface
st.title("Nepali Citizenship Document Scanner")
st.write("Upload both sides of your citizenship document")

col1, col2 = st.columns(2)
with col1:
    front_file = st.file_uploader("Front side", type=["jpg", "jpeg", "png"], key="front")
with col2:
    back_file = st.file_uploader("Back side", type=["jpg", "jpeg", "png"], key="back")

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
            labeled_front, flag_front, aspect_front = detect_and_label_citizenship_document(front_image, side="front")
            labeled_back, flag_back, aspect_back = detect_and_label_citizenship_document(back_image, side="back")

            st.subheader("Labeled Documents")
            col1, col2 = st.columns(2)
            with col1:
                st.image(labeled_front, caption="Labeled Front Side", use_container_width=True)
            with col2:
                st.image(labeled_back, caption="Labeled Back Side", use_container_width=True)

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
                    st.markdown(formatted_text)

                    st.download_button(
                        "Download as Text",
                        formatted_text,
                        file_name="citizenship_data.txt",
                    )
                else:
                    st.error("No information could be extracted")
            except Exception as e:
                st.error(f"Error: {str(e)}")