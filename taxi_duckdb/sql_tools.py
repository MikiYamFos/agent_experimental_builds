import os
import urllib.request

import duckdb

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
)
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"

con = duckdb.connect("taxi.db")


def setup_database():
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
    """)
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count


class SQLTools:
    def __init__(self) -> None:
        setup_database()

    def get_schema(self) -> str:
        """Runs DESCRIBE trips and returns all column names with their types."""
        rows = con.execute("DESCRIBE trips").fetchall()
        lines = ["column_name | column_type"]
        for row in rows:
            lines.append(f"{row[0]} | {row[1]}")
        return "\n".join(lines)

    def run_sql(self, query: str) -> str:
        """Executes a SQL query and returns column headers plus up to 50 rows."""
        limited_query = f"SELECT * FROM ({query.rstrip(';')}) AS results LIMIT 50"
        result = con.execute(limited_query)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        lines = [" | ".join(columns)]
        for row in rows:
            lines.append(" | ".join(str(value) for value in row))
        return "\n".join(lines)
