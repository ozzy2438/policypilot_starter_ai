"""
AFCA Datacube Ingestion
-----------------------
Reads the AFCA complaints CSV, cleans it, and loads into PostgreSQL.
Prints summary statistics for data exploration.

Usage:
    python -m app.ingest.load_afca          # full pipeline (requires Postgres)
    python -m app.ingest.load_afca --dry-run  # just show stats, no DB write
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# ── Setup ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

settings = get_settings()

# ── Schema ───────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS complaints (
    complaint_id        TEXT PRIMARY KEY,
    date_received       DATE,
    date_closed         DATE,
    product             TEXT,
    issue               TEXT,
    sub_issue           TEXT,
    company_type        TEXT,
    state               TEXT,
    complaint_text      TEXT,
    resolution          TEXT,
    resolution_days     INTEGER,
    compensation_aud    NUMERIC(12, 2)
);
"""


def load_csv() -> pd.DataFrame:
    """Read and clean the AFCA complaints CSV."""
    csv_path = settings.data_raw / "afca_complaints_sample.csv"
    log.info("Reading %s", csv_path)

    df = pd.read_csv(csv_path, parse_dates=["date_received", "date_closed"])

    # Basic cleaning
    df["complaint_text"] = df["complaint_text"].str.strip()
    df["product"] = df["product"].str.strip()
    df["issue"] = df["issue"].str.strip()
    df["resolution_days"] = pd.to_numeric(df["resolution_days"], errors="coerce")
    df["compensation_aud"] = pd.to_numeric(df["compensation_aud"], errors="coerce").fillna(0)

    log.info("Loaded %d complaints", len(df))
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print exploratory data analysis summary."""
    log.info("=" * 60)
    log.info("AFCA Datacube – Summary Statistics")
    log.info("=" * 60)

    # Product distribution
    log.info("\n📊 Complaints by Product:")
    product_counts = df["product"].value_counts()
    for product, count in product_counts.items():
        pct = count / len(df) * 100
        log.info("   %-25s %4d  (%5.1f%%)", product, count, pct)

    # Resolution outcomes
    log.info("\n📋 Resolution Outcomes:")
    resolution_counts = df["resolution"].value_counts()
    for resolution, count in resolution_counts.items():
        pct = count / len(df) * 100
        log.info("   %-45s %4d  (%5.1f%%)", resolution, count, pct)

    # State distribution
    log.info("\n🗺️  Complaints by State:")
    state_counts = df["state"].value_counts()
    for state, count in state_counts.items():
        pct = count / len(df) * 100
        log.info("   %-10s %4d  (%5.1f%%)", state, count, pct)

    # Timing stats
    log.info("\n⏱️  Resolution Time (days):")
    log.info("   Mean:   %.1f", df["resolution_days"].mean())
    log.info("   Median: %.1f", df["resolution_days"].median())
    log.info("   Min:    %d", df["resolution_days"].min())
    log.info("   Max:    %d", df["resolution_days"].max())

    # Compensation stats
    has_comp = df[df["compensation_aud"] > 0]
    log.info("\n💰 Compensation (AUD):")
    log.info("   Cases with compensation: %d / %d", len(has_comp), len(df))
    if len(has_comp) > 0:
        log.info("   Mean:   $%s", f'{has_comp["compensation_aud"].mean():,.0f}')
        log.info("   Median: $%s", f'{has_comp["compensation_aud"].median():,.0f}')
        log.info("   Total:  $%s", f'{has_comp["compensation_aud"].sum():,.0f}')


def write_to_postgres(df: pd.DataFrame) -> None:
    """Write DataFrame to PostgreSQL complaints table."""
    log.info("Connecting to PostgreSQL at %s:%s", settings.POSTGRES_HOST, settings.POSTGRES_PORT)

    conn = psycopg2.connect(settings.postgres_dsn)
    cur = conn.cursor()

    # Create table
    cur.execute(CREATE_TABLE)
    conn.commit()

    # Upsert rows
    cols = [
        "complaint_id", "date_received", "date_closed", "product",
        "issue", "sub_issue", "company_type", "state",
        "complaint_text", "resolution", "resolution_days", "compensation_aud",
    ]
    values = [tuple(row[c] for c in cols) for _, row in df.iterrows()]

    insert_sql = f"""
        INSERT INTO complaints ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (complaint_id) DO UPDATE SET
            complaint_text = EXCLUDED.complaint_text,
            resolution = EXCLUDED.resolution,
            resolution_days = EXCLUDED.resolution_days,
            compensation_aud = EXCLUDED.compensation_aud
    """
    execute_values(cur, insert_sql, values)
    conn.commit()

    log.info("✅ Wrote %d rows to PostgreSQL (complaints table)", len(values))

    cur.close()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load AFCA complaints into PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Only show stats, skip DB write")
    args = parser.parse_args()

    df = load_csv()
    print_summary(df)

    if not args.dry_run:
        write_to_postgres(df)
    else:
        log.info("🔸 Dry-run mode – skipping database write")

    # Save cleaned CSV to processed/
    processed_path = settings.data_processed / "afca_complaints_clean.csv"
    df.to_csv(processed_path, index=False)
    log.info("Saved cleaned data to %s", processed_path)


if __name__ == "__main__":
    main()
