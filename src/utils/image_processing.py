import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io


def preprocess_image(image_bytes):
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

    if is_front:
        field_positions = calculate_field_positions(width, height)
    else:
        field_positions = calculate_back_field_positions(width, height)

    try:
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
            "label_pos": (int(410), int(360)),
            "box_start": (
                int(410),
                int(380),
            ),
            "box_end": (int(980), int(430)),
        },
        
        "Father's Name": {
            "label_pos": (int(410), int(700)),
            "box_start": (
                int(410),
                int(720),
            ),
            "box_end": (int(980), int(765)),
        },
        "Mother's Name": {
            "label_pos": (int(410), int(800)),
            "box_start": (
                int(410),
                int(820),
            ),
            "box_end": (int(1100), int(870)),
        },
        "Date of Birth": {
            "label_pos": (int(410), int(630)),
            "box_start": (
                int(410),
                int(650),
            ),
            "box_end": (int(1250), int(700)),
        },
        "Permanent Address": {
            "label_pos": (int(410), int(515)),
            "box_start": (
                int(410),
                int(535),
            ),
            "box_end": (int(1470), int(645)),
        },
        "Citizenship Number": {
            "label_pos": (int(190), int(300)),
            "box_start": (
                int(190),
                int(320),
            ),
            "box_end": (int(490), int(370)),
        },
        "Gender":{
            "label_pos": (int(1250), int(360)),
            "box_start": (
                int(1250),
                int(380),
            ),
            "box_end": (int(1490), int(440)),
        }
        ,
        "Birth place": {
            "label_pos": (int(410), int(420)),
            "box_start": (
                int(410),
                int(440),
            ),
            "box_end": (int(990), int(530)),
        },
        "spouse": {
            "label_pos": (int(410), int(920)),
            "box_start": (
                int(410),
                int(930),
            ),
            "box_end": (int(900), int(980)),
        }
    }


def calculate_back_field_positions(width, height):
    """Calculate fixed positions for back side fields."""
    padding = 0.01

    return {
        "Issue Date": {
            "label_pos": (int(830), int(840)),
            "box_start": (
                int(830),
                int(860),
            ),
            "box_end": (int(1190), int(920)),
        },
        "Issuing Authority": {
            "label_pos": (int(830), int(900)),
            "box_start": (
                int(830),
                int(920),
            ),
            "box_end": (int(1490), int(1050)),
        }
    }
