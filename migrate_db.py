import sqlite3
import os

DB_PATH = os.path.join("backend", "sri_lankan_allergen_detector.db")

def migrate():
    print(f"Migrating {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print("Database file not found. SQLAlchemy will create it fresh.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    columns_to_add = [
        "work_env VARCHAR",
        "family_has_history BOOLEAN",
        "family_asthma BOOLEAN",
        "family_eczema BOOLEAN",
        "family_allergies VARCHAR"
    ]
    
    for col in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col}")
            print(f"Added column {col}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"Column {col.split(' ')[0]} already exists.")
            else:
                print(f"Error adding {col}: {e}")
                
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
