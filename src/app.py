import streamlit as st
import os
import json
import sys
from utils.data_extraction import extract_citizenship_data
from utils.image_processing import preprocess_image, label_citizenship_fields
from dotenv import load_dotenv
import io
from PIL import Image
from datetime import datetime

# Add parent directory to path to import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from database import CitizenshipDatabase

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Nepali Citizenship Data Extractor", page_icon="📄", layout="wide"
)

# Initialize database connection
db = CitizenshipDatabase()

# Add to session state if not already there
if 'data_saved' not in st.session_state:
    st.session_state.data_saved = False


def main():
    st.title("Nepali Citizenship Document Data Extractor")
    st.markdown(
        "Upload images of the front and back sides of the Nepali citizenship document to extract information."
    )

    # File uploaders that store bytes in session state
    front_uploaded_file = st.file_uploader(
        "Choose the front image...", type=["jpg", "jpeg", "png"], key="front"
    )
    if front_uploaded_file is not None:
        st.session_state.front_image_bytes = front_uploaded_file.getvalue()

    back_uploaded_file = st.file_uploader(
        "Choose the back image...", type=["jpg", "jpeg", "png"], key="back"
    )
    if back_uploaded_file is not None:
        st.session_state.back_image_bytes = back_uploaded_file.getvalue()

    # Check session state for images instead of uploaders
    if 'front_image_bytes' in st.session_state and 'back_image_bytes' in st.session_state:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(
            ["Original Images", "Preprocessed Images", "Labeled Fields"]
        )

        # Tab 1: Original images
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                front_image = Image.open(io.BytesIO(st.session_state.front_image_bytes))
                st.image(front_image, caption="Front Side", use_container_width=True)
            with col2:
                back_image = Image.open(io.BytesIO(st.session_state.back_image_bytes))
                st.image(back_image, caption="Back Side", use_container_width=True)

        # Tab 2: Preprocessed images
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                front_preprocessed = preprocess_image(st.session_state.front_image_bytes)
                st.image(
                    front_preprocessed,
                    caption="Preprocessed Front Side",
                    use_container_width=True,
                )

                # Store preprocessed image for OCR
                buffered = io.BytesIO()
                front_preprocessed.save(buffered, format="JPEG")
                st.session_state.front_preprocessed_bytes = buffered.getvalue()

            with col2:
                back_preprocessed = preprocess_image(st.session_state.back_image_bytes)
                st.image(
                    back_preprocessed,
                    caption="Preprocessed Back Side",
                    use_container_width=True,
                )

                # Store preprocessed image for OCR
                buffered = io.BytesIO()
                back_preprocessed.save(buffered, format="JPEG")
                st.session_state.back_preprocessed_bytes = buffered.getvalue()

        # Tab 3: Labeled fields
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                front_labeled = label_citizenship_fields(
                    st.session_state.front_image_bytes, is_front=True
                )
                st.image(
                    front_labeled,
                    caption="Labeled Front Side",
                    use_container_width=True,
                )
            with col2:
                back_labeled = label_citizenship_fields(
                    st.session_state.back_image_bytes, is_front=False
                )
                st.image(
                    back_labeled, caption="Labeled Back Side", use_container_width=True
                )

        # Process button to start extraction
        if st.button("Extract Information"):
            # Use preprocessed images for better extraction
            with st.spinner("Extracting information..."):
                front_result = extract_citizenship_data(
                    st.session_state.front_preprocessed_bytes, is_front=True
                )
                back_result = extract_citizenship_data(
                    st.session_state.back_preprocessed_bytes, is_front=False
                )

            if front_result and back_result:
                try:
                    # Clean response and convert to JSON
                    front_json_str = (
                        front_result.strip().replace("```json", "").replace("```", "")
                    )
                    back_json_str = (
                        back_result.strip().replace("```json", "").replace("```", "")
                    )

                    front_data = json.loads(front_json_str)
                    back_data = json.loads(back_json_str)

                    # Merge the data from both sides and add scan date
                    combined_data = {**front_data, **back_data}
                    combined_data["scan_date"] = datetime.now().strftime("%Y-%m-%d")

                    # Store data in session state
                    st.session_state.extracted_data = combined_data

                    # Display results in organized layout
                    st.success("Data extracted successfully!")

                    # Create columns layout for displaying extracted data
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Personal Information")
                        st.markdown(
                            f"**Full Name:** {combined_data.get('full_name', 'N/A')}"
                        )
                        st.markdown(
                            f"**Citizenship Number:** {combined_data.get('citizenship_no', 'N/A')}"
                        )
                        st.markdown(
                            f"**Date of Birth:** {combined_data.get('dob', 'N/A')}"
                        )
                        st.markdown(f"**Gender:** {combined_data.get('gender', 'N/A')}")
                        st.markdown(
                            f"**Birth Place:** {combined_data.get('birth_place', 'N/A')}"
                        )

                    with col2:
                        st.subheader("Family & Document Details")
                        st.markdown(
                            f"**Father's Name:** {combined_data.get('father_name', 'N/A')}"
                        )
                        st.markdown(
                            f"**Mother's Name:** {combined_data.get('mother_name', 'N/A')}"
                        )
                        st.markdown(
                            f"**Address:** {combined_data.get('permanent_address', 'N/A')}"
                        )
                        st.markdown(
                            f"**Issue Date:** {combined_data.get('issue_date', 'N/A')}"
                        )
                        st.markdown(
                            f"**Issuing Authority:** {combined_data.get('authority', 'N/A')}"
                        )
                        st.markdown(
                            f"**Spouse's Name:** {combined_data.get('spouse_name', 'N/A')}"
                        )

                    # Automatically save to database (without button)
                    save_status = st.empty()
                    
                    with save_status.container():
                        with st.spinner("Saving data to database..."):
                            try:
                                # Create a simple copy of the data
                                db_data = combined_data.copy()
                                
                                # Ensure proper date format for database
                                db_data["scan_date"] = datetime.now()
                                
                                # Make sure citizenship_no exists to avoid uniqueness issues
                                if not db_data.get("citizenship_no"):
                                    db_data["citizenship_no"] = f"AUTO-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                                # Direct database insertion to avoid potential issues with the class
                                import psycopg2
                                try:
                                    conn = psycopg2.connect(
                                        host=db.DB_HOST,
                                        database=db.DB_NAME,
                                        user=db.DB_USER,
                                        password=db.DB_PASS,
                                        port=db.DB_PORT
                                    )
                                    cursor = conn.cursor()
                                    
                                    # Insert data with explicit field mapping
                                    cursor.execute("""
                                        INSERT INTO citizenship_records 
                                        (full_name, father_name, mother_name, gender, citizenship_no, 
                                        permanent_address, dob, birth_place, issue_date, authority, 
                                        spouse_name, scan_date)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                        RETURNING id
                                    """, (
                                        db_data.get('full_name', 'N/A'),
                                        db_data.get('father_name', 'N/A'),
                                        db_data.get('mother_name', 'N/A'),
                                        db_data.get('gender', 'N/A'),
                                        db_data.get('citizenship_no', 'N/A'),
                                        db_data.get('permanent_address', db_data.get('address', 'N/A')),
                                        db_data.get('dob', 'N/A'),
                                        db_data.get('birth_place', 'N/A'),
                                        db_data.get('issue_date', 'N/A'),
                                        db_data.get('authority', 'N/A'),
                                        db_data.get('spouse_name', 'N/A'),
                                        db_data.get('scan_date', datetime.now())
                                    ))
                                    
                                    record_id = cursor.fetchone()[0]
                                    conn.commit()
                                    cursor.close()
                                    conn.close()
                                    
                                    st.success(f"Data saved to database (Record ID: {record_id})")
                                    st.session_state.data_saved = True
                                    
                                    # Add download button for JSON
                                    json_str = json.dumps(combined_data, indent=4)
                                    st.download_button(
                                        label="Download JSON data",
                                        data=json_str,
                                        file_name=f"citizenship_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                                        mime="application/json",
                                    )
                                    
                                except Exception as dbe:
                                    st.error(f"Database error: {str(dbe)}")
                                    
                                    # Still offer download option even if DB save failed
                                    json_str = json.dumps(combined_data, indent=4)
                                    st.download_button(
                                        label="Download JSON data instead",
                                        data=json_str,
                                        file_name=f"citizenship_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                                        mime="application/json",
                                    )
                                
                            except Exception as e:
                                st.error(f"Error preparing data: {str(e)}")

                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse response: {str(e)}")
                    st.write("Front result:", front_result)
                    st.write("Back result:", back_result)

    # Add a tab to view database records
    st.markdown("---")
    st.subheader("Database Records")

    if st.button("View Saved Records"):
        records = db.get_all_records()

        if records:
            st.success(f"Found {len(records)} records in database")

            for record in records:
                record_id, name, citizenship_no, scan_date = record

                with st.expander(f"{name} - {citizenship_no} (Scanned: {scan_date})"):
                    if st.button("View Details", key=f"view_{record_id}"):
                        record_details = db.get_record_by_id(record_id)

                        if record_details:
                            st.write("### Personal Details")
                            st.write(f"**Full Name:** {record_details[1]}")
                            st.write(f"**Father's Name:** {record_details[2]}")
                            st.write(f"**Mother's Name:** {record_details[3]}")
                            st.write(f"**Gender:** {record_details[4]}")

                            st.write("### Document Details")
                            st.write(f"**Citizenship No:** {record_details[5]}")
                            st.write(f"**Address:** {record_details[6]}")
                            st.write(f"**Date of Birth:** {record_details[7]}")
                            st.write(f"**Place of Birth:** {record_details[8]}")
                            st.write(f"**Issue Date:** {record_details[9]}")
                            st.write(f"**Authority:** {record_details[10]}")
                            st.write(f"**Spouse Name:** {record_details[11]}")
                            st.write(f"**Scan Date:** {record_details[12]}")
        else:
            st.info("No citizenship records found in the database")


if __name__ == "__main__":
    main()