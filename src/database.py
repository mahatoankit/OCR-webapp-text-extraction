import os
import psycopg2
from psycopg2 import sql
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CitizenshipDatabase:
    # Database connection parameters
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "citizenshipdata")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "password")
    DB_PORT = os.getenv("DB_PORT", "5432")

    def __init__(self):
        """Initialize the database connection and create tables if needed"""
        self.conn = None
        self.init_db()

    def get_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(
                host=self.DB_HOST,
                database=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASS,
                port=self.DB_PORT,
            )
            return conn
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return None

    def init_db(self):
        """Initialize database and create table if it doesn't exist"""
        self.conn = self.get_connection()
        if not self.conn:
            print("Failed to initialize database")
            return

        cursor = self.conn.cursor()

        # Create table for citizenship records
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS citizenship_records (
                id SERIAL PRIMARY KEY,
                full_name VARCHAR(255),
                father_name VARCHAR(255),
                mother_name VARCHAR(255),
                gender VARCHAR(50),
                citizenship_no VARCHAR(100) UNIQUE,
                permanent_address TEXT,
                dob VARCHAR(100),
                birth_place VARCHAR(255),
                issue_date VARCHAR(100),
                authority VARCHAR(255),
                spouse_name VARCHAR(255),
                scan_date TIMESTAMP
            )
            """
        )

        self.conn.commit()
        cursor.close()
        self.conn.close()

    def save_record(self, data):
        """Save extracted citizenship data to PostgreSQL database with debugging"""
        print("Starting save_record method...")
        conn = None
        cursor = None

        try:
            # Create new connection for this operation
            print(f"Connecting to database: {self.DB_HOST}:{self.DB_PORT}")
            conn = psycopg2.connect(
                host=self.DB_HOST,
                database=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASS,
                port=self.DB_PORT,
            )
            print("Connection established successfully")

            cursor = conn.cursor()
            print("Cursor created")

            # Insert data using parameterized query
            query = """
            INSERT INTO citizenship_records 
            (full_name, father_name, mother_name, gender, citizenship_no, 
            permanent_address, dob, birth_place, issue_date, authority, 
            spouse_name, scan_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """            # Prepare values with explicit conversion
            values = (
                str(data.get("full_name", "N/A")),
                str(data.get("father_name", "N/A")),
                str(data.get("mother_name", "N/A")),
                str(data.get("gender", "N/A")),
                str(data.get("citizenship_no", "N/A")),
                str(data.get("permanent_address", "N/A")),
                str(data.get("dob", "N/A")),
                str(data.get("birth_place", "N/A")),
                str(data.get("issue_date", "N/A")),
                str(data.get("authority", "N/A")),
                str(data.get("spouse_name", "N/A")),
                data.get("scan_date", datetime.now()),
            )

            print(f"Executing query with values: {values}")
            cursor.execute(query, values)
            print("Query executed")

            # Get the ID of the inserted record
            record_id = cursor.fetchone()[0]
            print(f"Record inserted with ID: {record_id}")

            # Commit changes
            print("Committing transaction...")
            conn.commit()
            print("Transaction committed")

            return True, f"Record saved with ID: {record_id}"

        except psycopg2.IntegrityError as e:
            error_msg = f"Data integrity error: {e}"
            print(error_msg)
            if conn:
                conn.rollback()
            return False, error_msg
        except psycopg2.OperationalError as e:
            error_msg = f"Operational error: Check database connection. {e}"
            print(error_msg)
            if conn:
                conn.rollback()
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(error_msg)
            if conn:
                conn.rollback()
            return False, error_msg
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_all_records(self):
        """Retrieve basic info for all records"""
        conn = self.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, full_name, citizenship_no, scan_date 
                FROM citizenship_records
                ORDER BY scan_date DESC
                """
            )
            return cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving records: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()

    def get_record_by_id(self, record_id):
        """Retrieve full details of a specific record"""
        conn = self.get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM citizenship_records
                WHERE id = %s
                """,
                (record_id,),
            )
            return cursor.fetchone()
        except Exception as e:
            print(f"Error retrieving record {record_id}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()