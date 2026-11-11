import csv
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "data" / "sample_irish_properties.csv"
DATABASE_FILE = BASE_DIR / "data" / "properties.db"


def create_database():
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            county TEXT NOT NULL,
            floor_area REAL NOT NULL,
            bedrooms INTEGER NOT NULL,
            property_type TEXT NOT NULL,
            ber_rating TEXT NOT NULL,
            asking_price REAL NOT NULL,
            market_value REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS valuations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER,
            county TEXT NOT NULL,
            floor_area REAL NOT NULL,
            bedrooms INTEGER NOT NULL,
            property_type TEXT NOT NULL,
            ber_rating TEXT NOT NULL,
            asking_price REAL NOT NULL,
            estimated_value REAL NOT NULL,
            difference REAL NOT NULL,
            percentage_difference REAL NOT NULL,
            price_assessment TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (property_id) REFERENCES properties(id)
        )
    """)

    # Prevent duplicate records when the script is run again.
    cursor.execute("DELETE FROM valuations")
    cursor.execute("DELETE FROM properties")
    cursor.execute(
        "DELETE FROM sqlite_sequence WHERE name IN ('properties', 'valuations')"
    )

    with open(CSV_FILE, mode="r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            cursor.execute("""
                INSERT INTO properties (
                    county,
                    floor_area,
                    bedrooms,
                    property_type,
                    ber_rating,
                    asking_price,
                    market_value
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["county"],
                float(row["floor_area"]),
                int(row["bedrooms"]),
                row["property_type"],
                row["ber_rating"],
                float(row["asking_price"]),
                float(row["market_value"])
            ))

    connection.commit()

    cursor.execute("SELECT COUNT(*) FROM properties")
    property_count = cursor.fetchone()[0]

    connection.close()

    print("Database created successfully")
    print(f"Imported properties: {property_count}")
    print(f"Database location: {DATABASE_FILE}")


if __name__ == "__main__":
    create_database()
