# Nepali Citizenship OCR Data Extractor

An AI-powered application that extracts information from Nepali citizenship documents using image processing and Google's Gemini model.

## Features

- Upload and process both front and back sides of Nepali citizenship documents
- Advanced image preprocessing for better text extraction
- Automatic field detection and labeling
- Data extraction using Google's Gemini AI model
- PostgreSQL database integration for data storage
- User-friendly interface built with Streamlit
- Export functionality for extracted data

## Technology Stack

- Python 3.x
- Streamlit
- OpenCV
- Google Gemini AI
- PostgreSQL
- PIL (Python Imaging Library)
- psycopg2
- python-dotenv

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Citizenship_OCR
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:

```
GOOGLE_API_KEY= <your gemini-2.0-flash api key>
DB_HOST=localhost
DB_NAME=citizenshipdata
DB_USER=postgres
DB_PASS= <database password>
DB_PORT=5432
```

## Database Setup

1. Ensure PostgreSQL is installed on your system
2. Navigate to the database directory:

```bash
cd database
```

3. Run the setup script:

```bash
psql -U postgres -f setup.sql
```

4. Verify the setup:

```bash
psql -U postgres -d citizenshipdata -c "\dt"
```

You should see the `citizenship_records` table listed.

## Project Structure

```
Citizenship_OCR/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── database.py         # Database operations
│   ├── utils/
│   │   ├── data_extraction.py    # AI extraction logic
│   │   ├── image_processing.py   # Image preprocessing
│   ├── config/
│   │   └── settings.py     # Configuration settings
├── requirements.txt
└── README.md
```

## Usage

1. Start the application:

```bash
cd src
streamlit run app.py
```

2. Upload citizenship document images:

   - Front side of the citizenship
   - Back side of the citizenship

3. View different processing stages:

   - Original Images
   - Preprocessed Images
   - Labeled Fields

4. Extract information using the "Extract Information" button

5. View and download the extracted data:
   - Data is automatically saved to the database
   - Option to download as JSON
   - View all saved records

## Features in Detail

### Image Processing

- Grayscale conversion
- Adaptive thresholding
- Noise removal
- Edge enhancement
- Field detection and labeling(using fixed co-ordinates)

### Data Extraction

- Personal information
- Document details
- Family information
- Address details
- Issue dates and authority information

### Database Operations

- Automatic data storage
- Record viewing and retrieval
- Unique citizenship number tracking
- Timestamp-based record management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
