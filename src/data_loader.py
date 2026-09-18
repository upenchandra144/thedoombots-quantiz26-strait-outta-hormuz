from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Centralized source path: this is the ONLY place where the raw Excel filename
# is defined. The workbook itself is never written to or modified by the app.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "R2-WAR ROOM MASTERPLAN.xlsx"
SHEET_NAME = "Shipment_Data"

ROUTE_MAP = {
    "Direct (Pre-Blockade)": "Direct",
    "Cape of Good Hope": "Cape",
    "Pipeline Bypass": "Pipeline",
    "Overland Truck": "Overland",
    "Air Bridge": "Air",
    "Held in Gulf": "Held",
}

EXPECTED_COLUMNS: List[str] = [
    "Shipment_ID",
    "Departure_Date",
    "Route_Type",
    "Product_Category",
    "Cargo_Type",
    "Customer_Name",
    "Customer_Region",
    "Customer_Since",
    "Cargo_Weight_Tons",
    "Cargo_Value_USD",
    "Contracted_Freight_Revenue_USD",
    "Planned_Transit_Days",
    "Actual_Transit_Days",
    "Delay_Days",
    "Freight_Cost_USD",
    "Fuel_Cost_USD",
    "Insurance_Cost_USD",
    "Penalty_Cost_USD",
    "Total_Cost_to_Serve_USD",
    "Revenue_Recognized_USD",
    "Gross_Margin_USD",
    "Gross_Margin_Pct",
    "DIFOT_Met",
    "Cost_per_Ton_USD",
    "Revenue_per_Ton_USD",
    "Route_Margin_Sensitivity_USD",
    "Customer_Concentration_Risk_Pct",
    "War_Risk_Insurance_Burden_Pct",
    "Delay_Cost_Attribution_Pct",
]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and normalize the shipment dataset in memory.

    The source workbook is read-only from the application's perspective.
    No save/write operation is performed against DATA_PATH.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Shipment dataset not found at: {DATA_PATH}. "
            "Please check data/raw/."
        )

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, engine="openpyxl")

    # Clean headers in memory only.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # Normalize date and categorical fields in memory.
    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")
    df["Route_Type"] = df["Route_Type"].astype("string").str.strip()
    df["Product_Category"] = df["Product_Category"].astype("string").str.strip()
    df["Customer_Name"] = df["Customer_Name"].astype("string").str.strip()
    df["Customer_Region"] = (
        df["Customer_Region"]
        .astype("string")
        .str.strip()
        .str.title()
    )
    df["DIFOT_Met"] = df["DIFOT_Met"].astype("string").str.strip().str.upper()

    # Reusable analytical fields. These are derived in memory and never saved
    # back to the source workbook.
    df["Route_Canonical"] = df["Route_Type"].map(ROUTE_MAP).fillna(df["Route_Type"])
    df["DIFOT_Flag"] = df["DIFOT_Met"].eq("Y")
    df["DIFOT_Pct"] = df["DIFOT_Flag"].astype(float) * 100.0
    df["Is_Held"] = df["Route_Canonical"].eq("Held")
    df["Is_Delivered"] = ~df["Is_Held"]

    # Avoid divide-by-zero surprises in downstream pages.
    df["Margin_per_Ton_USD"] = df["Gross_Margin_USD"].div(
        df["Cargo_Weight_Tons"].replace(0, pd.NA)
    )

    return df


def validate_data(df: pd.DataFrame) -> Dict[str, object]:
    """Return lightweight validation results used by the foundation app."""
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    duplicate_ids = int(df["Shipment_ID"].duplicated().sum()) if "Shipment_ID" in df else -1
    held_count = int(df["Is_Held"].sum()) if "Is_Held" in df else -1
    held_missing_actual = int(
        df.loc[df["Is_Held"], "Actual_Transit_Days"].isna().sum()
    ) if "Is_Held" in df else -1

    return {
        "rows": int(len(df)),
        "unique_shipment_ids": int(df["Shipment_ID"].nunique()) if "Shipment_ID" in df else -1,
        "duplicate_shipment_ids": duplicate_ids,
        "held_shipments": held_count,
        "held_missing_actual_transit": held_missing_actual,
        "missing_expected_columns": missing_columns,
        "dataset_ready": (
            len(df) > 0
            and not missing_columns
            and duplicate_ids == 0
        ),
    }


def source_status() -> Dict[str, object]:
    """Return source-file metadata without modifying the workbook."""
    return {
        "path": str(DATA_PATH),
        "exists": DATA_PATH.exists(),
        "filename": DATA_PATH.name,
    }
