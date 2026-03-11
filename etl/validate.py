"""
etl/validate.py
---------------
Step 3 of ETL Pipeline: Data Quality & Validation
- Checks for nulls, outliers, schema drift
- Generates a validation report
- Saves validated data to data/validated/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

VALIDATED_DIR = Path("data/validated")
VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("docs")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Schema Definition (expected columns + types)
# ─────────────────────────────────────────────
EXPECTED_SCHEMA = {
    "Store": "int64",
    "DayOfWeek": "int64",
    "Sales": "int64",
    "Customers": "int64",
    "Open": "int64",
    "Promo": "int64",
    "StateHoliday": "object",
    "SchoolHoliday": "int64",
}

REQUIRED_COLUMNS = list(EXPECTED_SCHEMA.keys())


class ValidationReport:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0

    def add(self, check_name: str, passed: bool, detail: str = ""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.checks.append({"check": check_name, "status": status, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"   {status} | {check_name}" + (f" → {detail}" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"  Validation Summary: {self.passed}/{total} checks passed")
        print(f"{'='*50}")
        return self.passed == total

    def save(self, filename: str = None):
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{ts}.txt"
        out = REPORT_DIR / filename
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"Data Validation Report — {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            for c in self.checks:
                f.write(f"{c['status']} | {c['check']}\n")
                if c['detail']:
                    f.write(f"         {c['detail']}\n")
            f.write(f"\nResult: {self.passed}/{self.passed + self.failed} passed\n")
        print(f"📄 Validation report saved → {out}")


# ─────────────────────────────────────────────
# Validation Checks
# ─────────────────────────────────────────────
def check_required_columns(df: pd.DataFrame, report: ValidationReport):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.add(
        "Required columns present",
        len(missing) == 0,
        f"Missing: {missing}" if missing else ""
    )


def check_no_nulls_in_critical(df: pd.DataFrame, report: ValidationReport):
    critical = ["Store", "Sales", "Date"]
    for col in critical:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            report.add(
                f"No nulls in '{col}'",
                null_count == 0,
                f"{null_count} nulls found" if null_count else ""
            )


def check_sales_non_negative(df: pd.DataFrame, report: ValidationReport):
    if "Sales" in df.columns:
        neg = (df["Sales"] < 0).sum()
        report.add("Sales values non-negative", neg == 0, f"{neg} negative values" if neg else "")


def check_sales_outliers(df: pd.DataFrame, report: ValidationReport):
    if "Sales" not in df.columns:
        return
    Q1 = df["Sales"].quantile(0.25)
    Q3 = df["Sales"].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    outliers = (df["Sales"] > upper).sum()
    pct = outliers / len(df) * 100
    report.add(
        "Sales outliers < 1% of data",
        pct < 1.0,
        f"{outliers:,} outliers ({pct:.2f}%)"
    )


def check_date_range(df: pd.DataFrame, report: ValidationReport):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        report.add(
            "Date range valid",
            min_date.year >= 2013,
            f"Range: {min_date.date()} → {max_date.date()}"
        )


def check_store_ids(df: pd.DataFrame, report: ValidationReport):
    if "Store" in df.columns:
        invalid = (df["Store"] <= 0).sum()
        report.add("All Store IDs positive", invalid == 0, f"{invalid} invalid IDs" if invalid else "")


def check_row_count(df: pd.DataFrame, report: ValidationReport, min_rows: int = 1000):
    report.add(
        f"Row count ≥ {min_rows:,}",
        len(df) >= min_rows,
        f"Got {len(df):,} rows"
    )


# ─────────────────────────────────────────────
# Main validation runner
# ─────────────────────────────────────────────
def run_validation(df: pd.DataFrame, save_validated: bool = True) -> bool:
    print("\n🔍 Running data validation checks...\n")
    report = ValidationReport()

    check_row_count(df, report)
    check_required_columns(df, report)
    check_no_nulls_in_critical(df, report)
    check_sales_non_negative(df, report)
    check_sales_outliers(df, report)
    check_date_range(df, report)
    check_store_ids(df, report)

    all_passed = report.summary()
    report.save()

    if save_validated and all_passed:
        out = VALIDATED_DIR / "validated_train.csv"
        df.to_csv(out, index=False)
        print(f"💾 Validated data saved → {out}")
    elif not all_passed:
        print("⚠️  Validation failed. Fix issues before loading to DB.")

    return all_passed


if __name__ == "__main__":
    df = pd.read_csv("data/processed/cleaned_train.csv")
    run_validation(df)
