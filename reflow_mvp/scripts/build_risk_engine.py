#!/usr/bin/env python3
"""
ReFlow AI — At-Risk Engine (Phase 2)

PURPOSE:
    Builds the "At-Risk Ledger" by applying the Shelf-Life Gatekeeper logic to
    lot-level inventory. Identifies lots that would be rejected at dock by the
    strictest customers (e.g., UNFI 75% RSL) but still have shelf life remaining—
    enabling FEFO rerouting to Discount Partner or B2B liquidation.

FMCG / RSL MATH:
    RSL (Remaining Shelf Life) = (expiry_date - today) / (expiry_date - production_date)
    = actual_days_remaining / standard_shelf_life_days.
    At receipt: RSL_at_receipt = (actual_days_remaining - transit_lead_time) / shelf_life.
    A lot is ELIGIBLE for customer C iff RSL_at_receipt >= C.required_rsl_pct.
    Is_At_Risk = ineligible for strictest customer AND actual_days_remaining > 0
    (i.e., commercially dead for some customers, not yet expired).

READS:  data/raw_synthetic/ (sku_master, customer_policies, lot_ledger)
WRITES: data/processed/at_risk_ledger.csv
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_synthetic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ingest_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Phase 1 CSVs from data/raw_synthetic/.

    Returns:
        (sku_master, customer_policies, lot_ledger). Dates parsed to datetime.
    """
    sku_master = pd.read_csv(RAW_DIR / "sku_master.csv")
    customer_policies = pd.read_csv(RAW_DIR / "customer_policies.csv")
    lot_ledger = pd.read_csv(RAW_DIR / "lot_ledger.csv")

    # Parse dates
    lot_ledger["production_date"] = pd.to_datetime(lot_ledger["production_date"])
    lot_ledger["expiry_date"] = pd.to_datetime(lot_ledger["expiry_date"])

    return sku_master, customer_policies, lot_ledger


def compute_current_date(lot_ledger: pd.DataFrame) -> datetime:
    """
    Compute evaluation date ("today") for RSL calculations.

    FMCG logic: Simulates "30 days after last production run"—typical replenishment
    cycle. If that date is in the future, use actual today to avoid future-dated eval.

    Returns:
        datetime for RSL evaluation.
    """
    max_production = lot_ledger["production_date"].max()
    candidate = max_production + timedelta(days=30)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # If candidate is in the future, use today (avoids future-dated evaluation)
    if candidate.date() > today.date():
        return today
    return candidate


def merge_and_enrich(
    lot_ledger: pd.DataFrame,
    sku_master: pd.DataFrame,
    current_date: datetime,
) -> pd.DataFrame:
    """
    Merge lot ledger with SKU master and compute value + remaining shelf life.

    Args:
        lot_ledger, sku_master: Raw data.
        current_date: "Today" for RSL.

    Returns:
        Enriched DataFrame with total_lot_value, actual_days_remaining.
    """
    df = lot_ledger.merge(sku_master, on="sku_id", how="left")
    df["total_lot_value"] = df["qty_on_hand"] * df["unit_cost"]
    df["actual_days_remaining"] = (df["expiry_date"] - pd.Timestamp(current_date)).dt.days
    return df


def apply_shelf_life_gatekeeper(
    df: pd.DataFrame,
    customer_policies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Shelf-Life Gatekeeper: per-customer eligibility.

    RSL at receipt = (actual_days_remaining - transit_lead_time) / shelf_life.
    Eligible iff RSL_at_receipt >= required_rsl_pct. Creates boolean columns
    eligible_<customer_name>.

    Args:
        df: Enriched lot ledger.
        customer_policies: required_rsl_pct, transit_lead_time_days per customer.

    Returns:
        DataFrame with eligible_* columns added.
    """
    for _, row in customer_policies.iterrows():
        customer_name = row["customer_name"]
        # Sanitize for column name (e.g., "Rodriguez, Figueroa and Sanchez" -> "Rodriguez_Figueroa_and_Sanchez")
        col_name = "eligible_" + customer_name.replace(",", "").replace(" ", "_").lower()

        # RSL at receipt: deduct transit from "expiry clock" before comparing to policy
        rsl_at_receipt = (
            df["actual_days_remaining"] - row["transit_lead_time_days"]
        ) / df["standard_shelf_life_days"]
        df[col_name] = rsl_at_receipt >= row["required_rsl_pct"]

    return df


def flag_risk_and_exposure(
    df: pd.DataFrame,
    customer_policies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag at-risk lots and compute financial exposure.

    Is_At_Risk: Ineligible for strictest customer (e.g., UNFI 75%) but not expired.
    These lots can still be rerouted to Discount Partner (10% RSL) or liquidated.
    Financial_Risk_Exposure = Total_Lot_Value for at-risk lots, 0 otherwise.

    Returns:
        DataFrame with is_at_risk, financial_risk_exposure columns.
    """
    strictest = customer_policies.loc[customer_policies["required_rsl_pct"].idxmax()]
    strictest_name = strictest["customer_name"]
    strictest_col = "eligible_" + strictest_name.replace(",", "").replace(" ", "_").lower()

    df["is_at_risk"] = (~df[strictest_col]) & (df["actual_days_remaining"] > 0)
    df["financial_risk_exposure"] = df["total_lot_value"].where(df["is_at_risk"], 0)
    return df


def compute_at_risk_ledger(
    sku_master: pd.DataFrame,
    customer_policies: pd.DataFrame,
    lot_ledger: pd.DataFrame,
    current_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Run the full At-Risk Engine pipeline on DataFrames (no disk I/O).

    Use this for uploaded data in the Streamlit app or programmatic calls.

    Args:
        sku_master: Columns: sku_id, category, unit_cost, standard_shelf_life_days.
        customer_policies: Columns: customer_id, customer_name, required_rsl_pct,
            transit_lead_time_days.
        lot_ledger: Columns: lot_id, sku_id, location_id, qty_on_hand,
            production_date, expiry_date. Dates must be parseable.
        current_date: "Today" for RSL. If None, uses max(production_date) + 30.

    Returns:
        Enriched DataFrame with total_lot_value, actual_days_remaining,
        eligible_* columns, is_at_risk, financial_risk_exposure.
    """
    lot_ledger = lot_ledger.copy()
    lot_ledger["production_date"] = pd.to_datetime(lot_ledger["production_date"])
    lot_ledger["expiry_date"] = pd.to_datetime(lot_ledger["expiry_date"])

    if current_date is None:
        current_date = compute_current_date(lot_ledger)

    df = merge_and_enrich(lot_ledger, sku_master, current_date)
    df = apply_shelf_life_gatekeeper(df, customer_policies)
    df = flag_risk_and_exposure(df, customer_policies)
    return df


def main() -> None:
    """Build the At-Risk Ledger and save to data/processed/."""
    print("ReFlow AI — At-Risk Engine (Phase 2)")
    print("=" * 50)

    sku_master, customer_policies, lot_ledger = ingest_data()
    print(f"Ingested: {len(lot_ledger)} lots, {len(sku_master)} SKUs, {len(customer_policies)} customers")

    df = compute_at_risk_ledger(sku_master, customer_policies, lot_ledger)
    print("Flagged at-risk lots and Financial_Risk_Exposure")
    print()

    # 5. Output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "at_risk_ledger.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Summary
    n_at_risk = df["is_at_risk"].sum()
    total_exposure = df["financial_risk_exposure"].sum()
    total_value = df["total_lot_value"].sum()
    print()
    print("Summary")
    print("-" * 50)
    print(f"  Lots at risk (ineligible for strictest customer): {n_at_risk} / {len(df)}")
    print(f"  Total inventory value: ${total_value:,.2f}")
    print(f"  Financial risk exposure (rejection risk): ${total_exposure:,.2f}")
    print()
    print(f"Found ${total_exposure:,.2f} total inventory value at risk of rejection.")


if __name__ == "__main__":
    main()
