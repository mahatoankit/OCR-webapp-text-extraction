import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io


def preprocess_image(image_bytes):
    """
    Preprocess the image to enhance text visibility

    Args:
        image_bytes: Raw image bytes

    Returns:
        PIL Image: Preprocessed image
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Noise removal using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Edge enhancement
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    enhanced = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)

    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Convert to PIL image
    pil_img = Image.fromarray(enhanced_rgb)

    return pil_img


def convert_image_to_bytes(image):
    """Convert a PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # Save as PNG or any other format
    img_byte_arr.seek(0)  # Move to the beginning of the BytesIO buffer
    return img_byte_arr.getvalue()


def label_citizenship_fields(image_bytes, is_front=True):
    """
    Label the fields on citizenship image with more precise fixed coordinates.

    Args:
        image_bytes: Raw image bytes
        is_front: Boolean indicating if it's front side

    Returns:
        PIL Image: Image with labeled fields
    """
    # Convert bytes to PIL Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Apply basic preprocessing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    img_cv = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    width, height = img.size

    # Get field positions based on side (front/back)
    if is_front:
        field_positions = calculate_field_positions(width, height)
    else:
        field_positions = calculate_back_field_positions(width, height)

    # Draw boxes and labels for each field
    try:
        # Try to load a font, use default if not available
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for field_name, positions in field_positions.items():
        # Draw rectangle
        draw.rectangle(
            [positions["box_start"], positions["box_end"]], outline=(0, 200, 0), width=2
        )

        # Add text label
        draw.text(positions["label_pos"], field_name, fill=(255, 0, 0), font=font)

    return img


def calculate_field_positions(width, height):
    """Calculate fixed positions for front side fields."""
    # Padding to make boxes slightly larger than the text area
    padding = 0.01

    return {
        "Full Name": {
            "label_pos": (int(width * 0.1), int(height * 0.15)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.13 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.20 + padding))),
        },
        
        "Father's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.25)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.23 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.30 + padding))),
        },
        "Mother's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.35)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.33 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.40 + padding))),
        },
        "Date of Birth": {
            "label_pos": (int(width * 0.1), int(height * 0.45)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.43 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.50 + padding))),
        },
        "Place of Birth": {
            "label_pos": (int(width * 0.1), int(height * 0.55)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.53 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.60 + padding))),
        },
        "Citizenship Number": {
            "label_pos": (int(width * 0.1), int(height * 0.75)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.73 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.80 + padding))),
        },
        "Permanent Address": {
            "label_pos": (int(width * 0.1), int(height * 0.85)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.83 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.90 + padding))),
        },
    }


def calculate_back_field_positions(width, height):
    """Calculate fixed positions for back side fields."""
    padding = 0.01

    return {
        "Spouse's Name": {
            "label_pos": (int(width * 0.1), int(height * 0.15)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.13 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.25 + padding))),
        },
        "Issue Date": {
            "label_pos": (int(width * 0.1), int(height * 0.35)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.33 - padding)),
            ),
            "box_end": (int(width * (0.6 + padding)), int(height * (0.40 + padding))),
        },
        "Issuing Authority": {
            "label_pos": (int(width * 0.1), int(height * 0.55)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.53 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.65 + padding))),
        },
        "Office": {
            "label_pos": (int(width * 0.1), int(height * 0.70)),
            "box_start": (
                int(width * (0.25 - padding)),
                int(height * (0.68 - padding)),
            ),
            "box_end": (int(width * (0.9 + padding)), int(height * (0.75 + padding))),
        },
    }
