"""
tests/test_validate.py
Tests for the data validation module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.validate import (
    check_sales_non_negative,
    check_required_columns,
    check_row_count,
    ValidationReport
)


@pytest.fixture
def good_df():
    return pd.DataFrame({
        "Store": [1, 2, 3],
        "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
        "Sales": [5000, 7000, 4500],
        "Customers": [300, 450, 280],
        "Open": [1, 1, 1],
        "Promo": [1, 0, 1],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [0, 0, 1],
    })


@pytest.fixture
def bad_df():
    return pd.DataFrame({
        "Store": [1, 2],
        "Date": ["2015-01-01", "2015-01-02"],
        "Sales": [-100, 7000],  # negative sales
        "Customers": [300, 450],
        "Open": [1, 1],
        "Promo": [1, 0],
        "StateHoliday": ["0", "0"],
        "SchoolHoliday": [0, 0],
    })


def test_sales_non_negative_pass(good_df):
    report = ValidationReport()
    check_sales_non_negative(good_df, report)
    assert report.passed == 1
    assert report.failed == 0


def test_sales_non_negative_fail(bad_df):
    report = ValidationReport()
    check_sales_non_negative(bad_df, report)
    assert report.failed == 1


def test_required_columns_pass(good_df):
    report = ValidationReport()
    check_required_columns(good_df, report)
    assert report.passed == 1


def test_required_columns_fail():
    df = pd.DataFrame({"Store": [1], "Sales": [500]})  # missing columns
    report = ValidationReport()
    check_required_columns(df, report)
    assert report.failed == 1


def test_row_count(good_df):
    report = ValidationReport()
    check_row_count(good_df, report, min_rows=2)
    assert report.passed == 1
    check_row_count(good_df, report, min_rows=1000)
    assert report.failed == 1
