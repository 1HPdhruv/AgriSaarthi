import sqlite3
import pandas as pd

DB_PATH = "farmer_history.db"

def save_disease_history(
    farmer_name,
    crop,
    disease,
    remedy_en, precautions_en,
    remedy_hi, precautions_hi,
    remedy_ta, precautions_ta,
    remedy_te, precautions_te,
    remedy_ml, precautions_ml
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist, with columns for all 5 languages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS disease_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            farmer_name TEXT,
            crop TEXT,
            disease TEXT,
            remedy_en TEXT, precautions_en TEXT,
            remedy_hi TEXT, precautions_hi TEXT,
            remedy_ta TEXT, precautions_ta TEXT,
            remedy_te TEXT, precautions_te TEXT,
            remedy_ml TEXT, precautions_ml TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert new record
    cursor.execute("""
        INSERT INTO disease_history (
            farmer_name, crop, disease,
            remedy_en, precautions_en,
            remedy_hi, precautions_hi,
            remedy_ta, precautions_ta,
            remedy_te, precautions_te,
            remedy_ml, precautions_ml
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farmer_name, crop, disease,
        remedy_en, precautions_en,
        remedy_hi, precautions_hi,
        remedy_ta, precautions_ta,
        remedy_te, precautions_te,
        remedy_ml, precautions_ml
    ))
    
    conn.commit()
    conn.close()


def get_disease_history():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM disease_history ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df