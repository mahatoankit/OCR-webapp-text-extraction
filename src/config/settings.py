# filepath: /citizenship-data-extractor/citizenship-data-extractor/src/config/settings.py
API_KEY = "AIzaSyCotnndBVRVBcGlHUiKbP_I_mTmAr1Ages"
IMAGE_UPLOAD_LIMIT = 5  # in MB
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
EXTRACTION_PROMPT = """
Analyze this citizenship document image and extract the following information in JSON format:
{
    "name": "full name",
    "date_of_birth": "YYYY-MM-DD",
    "citizenship_number": "number",
    "issue_date": "YYYY-MM-DD",
    "father_name": "full name",
    "mother_name": "full name",
    "address": "full address"
}
Return only the JSON object without any additional text or formatting.
"""