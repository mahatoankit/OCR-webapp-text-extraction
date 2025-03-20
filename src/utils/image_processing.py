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
    Label the fields on citizenship image

    Args:
        image_bytes: Raw image bytes
        is_front: Boolean indicating if it's front side

    Returns:
        PIL Image: Image with labeled fields
    """
    # Convert bytes to PIL Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    # Define field positions based on relative coordinates
    # These are approximate and may need adjustment for different images
    width, height = img.size

    if is_front:
        fields = {
            "Full Name": (width * 0.5, height * 0.3),
            "Citizenship No": (width * 0.3, height * 0.2),
            "Date of Birth": (width * 0.7, height * 0.4),
            "Father's Name": (width * 0.5, height * 0.5),
            "Mother's Name": (width * 0.5, height * 0.6),
            "Issue Date": (width * 0.7, height * 0.7),
        }
    else:
        fields = {
            "Photo Area": (width * 0.2, height * 0.25),
            "Address": (width * 0.6, height * 0.3),
            "Issuing Authority": (width * 0.6, height * 0.5),
            "Spouse Name": (width * 0.5, height * 0.6),
            "Fingerprint": (width * 0.2, height * 0.7),
        }

    # Draw boxes and labels for each field
    for field_name, (x, y) in fields.items():
        # Calculate box dimensions
        box_width = width * 0.35
        box_height = height * 0.08

        # Draw rectangle
        rect_shape = [
            (x - box_width / 2, y - box_height / 2),
            (x + box_width / 2, y + box_height / 2),
        ]
        draw.rectangle(rect_shape, outline=(0, 255, 0), width=2)  # Green outline

        # Add text label
        draw.text(
            (x - box_width / 2 + 5, y - box_height / 2 - 15),
            field_name,
            fill=(255, 0, 0),
        )  # Red text

    return img
