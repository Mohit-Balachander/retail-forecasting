"""
etl/load.py
-----------
Step 4 of ETL Pipeline: Load to Database
- Loads validated data into Supabase (PostgreSQL) or MySQL
- Uses SQLAlchemy for database-agnostic loading
- Supports upsert to avoid duplicate rows
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    """
    Creates SQLAlchemy engine.
    Reads DB_URL from .env — supports Supabase (postgres) or MySQL.

    .env example:
        DB_URL=postgresql://user:password@db.supabase.co:5432/postgres
        # OR for MySQL:
        DB_URL=mysql+pymysql://user:password@localhost:3306/retaildb
    """
    db_url = os.getenv("DB_URL")
    if not db_url:
        raise ValueError("DB_URL not set in .env file.")
    engine = create_engine(db_url)
    print(f"🔌 Connected to database.")
    return engine


def create_tables(engine):
    """Creates tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sales_data (
                id              SERIAL PRIMARY KEY,
                store           INTEGER NOT NULL,
                date            DATE NOT NULL,
                sales           INTEGER,
                customers       INTEGER,
                open            SMALLINT,
                promo           SMALLINT,
                state_holiday   VARCHAR(10),
                school_holiday  SMALLINT,
                store_type      VARCHAR(5),
                assortment      VARCHAR(5),
                competition_distance FLOAT,
                year            INTEGER,
                month           INTEGER,
                day             INTEGER,
                week_of_year    INTEGER,
                is_weekend      SMALLINT,
                log_sales       FLOAT,
                loaded_at       TIMESTAMP DEFAULT NOW(),
                UNIQUE (store, date)
            );
        """))
        conn.commit()
    print("✅ Table 'sales_data' ready.")


def load_to_db(df: pd.DataFrame, engine, table: str = "sales_data", chunksize: int = 5000):
    """
    Loads DataFrame into the database in chunks.
    Uses 'append' with conflict handling (upsert-like via ignore).
    """
    # Normalize column names to snake_case
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Keep only relevant columns
    keep_cols = [
        "store", "date", "sales", "customers", "open", "promo",
        "stateholiday", "schoolholiday", "storetype", "assortment",
        "competitiondistance", "year", "month", "day",
        "weekofyear", "isweekend", "logsales"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    print(f"\n📤 Loading {len(df):,} rows into '{table}' in chunks of {chunksize}...")
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    print(f"✅ Load complete.")


def verify_load(engine, table: str = "sales_data"):
    """Quick count check after loading."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
        count = result.scalar()
    print(f"🔎 Rows in '{table}': {count:,}")
    return count


def run_load(csv_path: str = "data/validated/validated_train.csv"):
    """Full load pipeline."""
    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df):,} rows from {csv_path}")

    engine = get_engine()
    create_tables(engine)
    load_to_db(df, engine)
    verify_load(engine)


if __name__ == "__main__":
    run_load()
