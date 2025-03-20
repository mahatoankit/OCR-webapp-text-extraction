# Nepali Citizenship Document Scanner

This repository contains an OCR web application for extracting information from Nepali citizenship documents. It provides two implementations:

- **Google Gemini OCR**: Uses the Gemini model for advanced text extraction. (See `app.py`)
- **Tesseract OCR**: Uses Tesseract for local OCR processing. (See `tessApp.py`)

## Features

- **Automatic Orientation Correction**: Detects and rotates image based on text orientation.
- **Image Preprocessing**: Enhances contrast and reduces noise to improve OCR accuracy.
- **Region-Based Extraction**: Defines areas on the document for specific fields.
- **Multi-Language Support**: Extracts text in both Nepali and English.
- **Visual Feedback**: Highlights extraction regions on the processed images.
- **Downloadable Output**: Provides an option to download the extracted data as a text file.
  
## Installation

### Prerequisites

- Python 3.8 or higher
- On Linux, install required system packages:

```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-nep
```

### Python Dependencies

Install the required packages:

```bash
pip install streamlit pillow opencv-python numpy pytesseract python-dotenv google-generativeai
```

## Configuration

Create a `.env` file in the repository root to store your environment variables, for example:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

### Google Gemini OCR Application (`app.py`)

Run the application with:

```bash
streamlit run app.py
```

Upload both the front and back images of the citizenship document. The app will use the Google Gemini API to extract the relevant information.

### Tesseract OCR Application (`tessApp.py`)

For a local OCR solution powered by Tesseract:

```bash
streamlit run tessApp.py
```

This version is optimized solely for image-based processing using Tesseract and is ideal if you prefer a local approach or do not have access to the Gemini API.

## Repository Structure

```
OCR-webapp-text-extraction/
├── app.py             # Application using the Google Gemini API.
├── tessApp.py         # Application using Tesseract OCR.
├── .env               # Environment variables file (create manually).
├── requirements.txt   # Python dependencies (optional).
└── README.md          # This file.
```
