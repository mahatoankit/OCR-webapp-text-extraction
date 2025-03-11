import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import os
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


def preprocess_image(img):
    # Add contrast enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def calculate_field_positions(width, height):
    # Add some padding to field boxes
    padding = 0.02
    return {
        "नाम थर / Full Name": {
            "label_pos": (int(width * 0.1), int(height * 0.15)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.13 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.20 + padding))),
        },
        # ... adjust other fields similarly
    }


def detect_and_label_citizenship_document(image, side="front"):
    # Convert PIL Image to CV2 format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = preprocess_image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase kernel size for better noise reduction
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adjust Canny parameters
    edges = cv2.Canny(blurred, 30, 200)  # Modified thresholds

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

        back_fields = {
            "पति/पत्नीको नाम / Spouse's Name": {
                "label_pos": (int(width * 0.1), int(height * 0.15)),
                "box_start": (int(width * 0.25), int(height * 0.13)),
                "box_end": (int(width * 0.9), int(height * 0.20)),
            },
            "औंठा छाप / Fingerprints": {
                "label_pos": (int(width * 0.1), int(height * 0.35)),
                "box_start": (int(width * 0.25), int(height * 0.33)),
                "box_end": (int(width * 0.9), int(height * 0.50)),
            },
            "जारी गर्ने अधिकारी / Issuing Authority": {
                "label_pos": (int(width * 0.1), int(height * 0.65)),
                "box_start": (int(width * 0.25), int(height * 0.63)),
                "box_end": (int(width * 0.9), int(height * 0.80)),
            },
        }

        fields = front_fields if side == "front" else back_fields

        # Draw labels and boxes with unicode font support
        for field, positions in fields.items():
            # Draw the label with a better font
            img_pil = Image.fromarray(result)
            draw = ImageDraw.Draw(img_pil)

            # Use a TrueType font that supports Devanagari
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/unifont/unifont.ttf", 32
                )
            except:
                # Fallback to default font
                font = ImageFont.load_default()

            draw.text(positions["label_pos"], field, font=font, fill=(255, 0, 0))
            result = np.array(img_pil)

            # Draw box for the field area
            cv2.rectangle(
                result,
                positions["box_start"],
                positions["box_end"],
                (0, 0, 255),  # Red color for boxes
                2,
            )

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), True

    return image, False


# Update Streamlit interface
st.title("Nepali Citizenship Document scanner portal")
st.write("Upload both sides of your Nepali citizenship document")

# File uploaders for both sides
front_file = st.file_uploader(
    "Upload front side...", type=["jpg", "jpeg", "png"], key="front"
)
back_file = st.file_uploader(
    "Upload back side...", type=["jpg", "jpeg", "png"], key="back"
)

if front_file is not None and back_file is not None:
    # Process front side
    front_image = Image.open(front_file)
    st.write("Front Side:")
    st.image(front_image, caption="Front Side", use_container_width=True)

    processed_front, front_detected = detect_and_label_citizenship_document(
        front_image, "front"
    )

    # Process back side
    back_image = Image.open(back_file)
    st.write("Back Side:")
    st.image(back_image, caption="Back Side", use_container_width=True)

    processed_back, back_detected = detect_and_label_citizenship_document(
        back_image, "back"
    )

    if front_detected and back_detected:
        st.write("Detected Document Boundaries:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                processed_front, caption="Processed Front", use_container_width=True
            )
        with col2:
            st.image(processed_back, caption="Processed Back", use_container_width=True)

        st.write("Extracting information...")
        try:
            prompt = [
                "This is a Nepali citizenship document (both sides). Please extract and organize the following information:",
                "Front Side:",
                "1. नाम थर / Full Name",
                "2. बुवाको नाम / Father's Name",
                "3. आमाको नाम / Mother's Name",
                "4. जन्म मिति / Date of Birth",
                "5. जन्म स्थान / Place of Birth",
                "6. लिङ्ग / Gender",
                "7. नागरिकता नं / Citizenship Number",
                "8. स्थायी ठेगाना / Permanent Address",
                "9. जारी मिति / Issue Date",
                "Back Side:",
                "1. पति/पत्नीको नाम / Spouse's Name (if applicable)",
                "2. औंठा छाप / Fingerprint details",
                "3. जारी गर्ने अधिकारी / Issuing Authority",
                "Please format the information in both Nepali and English where available.",
                processed_front,
                processed_back,
            ]

            response = model.generate_content(prompt)
            if response and response.text:
                extracted_text = response.text
                # Validate extracted text contains expected fields
                required_fields = ["नाम थर", "नागरिकता नं", "जारी मिति"]
                if not all(field in extracted_text for field in required_fields):
                    st.warning(
                        "Some required fields could not be detected. Please check image quality."
                    )
                st.markdown("### Extracted Information:")
                st.markdown(extracted_text)
            else:
                st.error("No text could be extracted from the document.")
        except Exception as e:
            st.error(f"Error in text extraction: {str(e)}")
    else:
        st.error(
            "Could not detect document boundaries. Please ensure both images are clear and well-lit."
        )
