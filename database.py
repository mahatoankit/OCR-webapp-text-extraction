import psycopg2
from psycopg2 import sql
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class CitizenshipDatabase:
    # Database connection parameters
    DB_HOST = os.getenv("DB_HOST")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_PORT = os.getenv("DB_PORT")

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
            print(f"Database connection error: {e}")
            return None

    def init_db(self):
        """Initialize database and create table if it doesn't exist"""
        self.conn = self.get_connection()
        if not self.conn:
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
        """Save extracted citizenship data to PostgreSQL database"""
        conn = self.get_connection()
        if not conn:
            return False, "Database connection failed"

        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO citizenship_records (
                    full_name, father_name, mother_name, gender, 
                    citizenship_no, permanent_address, dob, birth_place,
                    issue_date, authority, spouse_name, scan_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    data.get("full_name", "Not provided"),
                    data.get("father_name", "Not provided"),
                    data.get("mother_name", "Not provided"),
                    data.get("gender", "Not provided"),
                    data.get("citizenship_no", "Not provided"),
                    data.get("address", "Not provided"),
                    data.get("dob", "Not provided"),
                    data.get("birth_place", "Not provided"),
                    data.get("issue_date", "Not provided"),
                    data.get("authority", "Not provided"),
                    data.get("spouse_name", "Not provided"),
                    datetime.now(),
                ),
            )

            conn.commit()
            return True, "Record saved successfully"
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            return False, "This citizenship number already exists in database"
        except Exception as e:
            conn.rollback()
            return False, f"Error saving record: {str(e)}"
        finally:
            cursor.close()
            conn.close()

    def get_all_records(self):
        conn = self.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, full_name, citizenship_no, scan_date FROM citizenship_records ORDER BY scan_date DESC"
            )
            records = cursor.fetchall()
            return records
        except Exception as e:
            print(f"Error retrieving records: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_record_by_id(self, record_id):
        """Retrieve full details of a specific record"""
        conn = self.get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM citizenship_records WHERE id = %s", (record_id,)
            )
            record = cursor.fetchone()
            return record
        except Exception as e:
            print(f"Error retrieving record {record_id}: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
