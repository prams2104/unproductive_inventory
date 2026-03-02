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
from typing import TYPE_CHECKING

import pandas as pd

from schema import ValidationResult, validate_dataframe, validate_referential_integrity

if TYPE_CHECKING:
    from fefo_engine import RerouteRecommendation

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_synthetic"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def ingest_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw Phase 1 CSVs from data/raw_synthetic/. Validates before return.

    Returns:
        (sku_master, customer_policies, lot_ledger). Dates parsed to datetime.
    Raises:
        ValueError: If validation fails (missing columns, zero valid rows).
    """
    sku_master = pd.read_csv(RAW_DIR / "sku_master.csv")
    customer_policies = pd.read_csv(RAW_DIR / "customer_policies.csv")
    lot_ledger = pd.read_csv(RAW_DIR / "lot_ledger.csv")

    sku_res = validate_dataframe(sku_master, "sku_master")
    if not sku_res.is_valid or sku_res.cleaned_df is None:
        raise ValueError(f"SKU Master validation failed: {[e.message for e in sku_res.errors]}")
    sku_master = sku_res.cleaned_df

    policy_res = validate_dataframe(customer_policies, "customer_policies")
    if not policy_res.is_valid or policy_res.cleaned_df is None:
        raise ValueError(f"Customer Policies validation failed: {[e.message for e in policy_res.errors]}")
    customer_policies = policy_res.cleaned_df

    lot_res = validate_dataframe(lot_ledger, "lot_ledger")
    if not lot_res.is_valid or lot_res.cleaned_df is None:
        raise ValueError(f"Lot Ledger validation failed: {[e.message for e in lot_res.errors]}")
    lot_ledger = lot_res.cleaned_df

    ref_res = validate_referential_integrity(lot_ledger, sku_master)
    if ref_res.cleaned_df is not None:
        lot_ledger = ref_res.cleaned_df

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


def build_comprehensive_ledger(
    sku_master: pd.DataFrame,
    customer_policies: pd.DataFrame,
    lot_ledger: pd.DataFrame,
    edi_852: pd.DataFrame | None = None,
    current_date: datetime | None = None,
    carrying_cost_pct: float = 0.25,
    daily_storage_cost_per_unit: float = 0.02,
) -> tuple[pd.DataFrame, list["RerouteRecommendation"], ValidationResult]:
    """
    Full pipeline: validate → enrich → gatekeeper → eligibility → FEFO → drain → signal.

    Args:
        sku_master, customer_policies, lot_ledger: Input DataFrames.
        edi_852: Optional EDI 852 for signal integrity scoring.
        current_date: "Today" for RSL. If None, uses max(production_date) + 30.
        carrying_cost_pct, daily_storage_cost_per_unit: For drain rate.

    Returns:
        (comprehensive_ledger, reroute_recommendations, validation_result).
    """
    from fefo_engine import build_eligibility_matrix, compute_urgency_score, recommend_reroutes
    from drain_rate import compute_drain_rate_ledger
    from signal_integrity import score_signal_integrity, summarize_signal_integrity

    # 1. Validate
    sku_res = validate_dataframe(sku_master, "sku_master")
    policy_res = validate_dataframe(customer_policies, "customer_policies")
    lot_res = validate_dataframe(lot_ledger, "lot_ledger")

    all_errors = sku_res.errors + policy_res.errors + lot_res.errors
    if not sku_res.is_valid or not policy_res.is_valid or not lot_res.is_valid:
        combined = ValidationResult(
            is_valid=False,
            errors=all_errors,
            warnings=sku_res.warnings + policy_res.warnings + lot_res.warnings,
            stats={},
        )
        return (pd.DataFrame(), [], combined)

    sku_clean = sku_res.cleaned_df
    policy_clean = policy_res.cleaned_df
    lot_clean = lot_res.cleaned_df
    if sku_clean is None or policy_clean is None or lot_clean is None:
        return (pd.DataFrame(), [], ValidationResult(is_valid=False, errors=all_errors, stats={}))

    ref_res = validate_referential_integrity(lot_clean, sku_clean)
    if ref_res.cleaned_df is not None:
        lot_clean = ref_res.cleaned_df

    combined_warnings = sku_res.warnings + policy_res.warnings + lot_res.warnings + ref_res.warnings

    # 2. Merge and enrich
    lot_clean = lot_clean.copy()
    lot_clean["production_date"] = pd.to_datetime(lot_clean["production_date"])
    lot_clean["expiry_date"] = pd.to_datetime(lot_clean["expiry_date"])
    if current_date is None:
        current_date = compute_current_date(lot_clean)

    df = merge_and_enrich(lot_clean, sku_clean, current_date)
    df = apply_shelf_life_gatekeeper(df, policy_clean)
    df = flag_risk_and_exposure(df, policy_clean)

    # 3. Eligibility matrix + FEFO recommendations
    em = build_eligibility_matrix(df, policy_clean)
    recommendations = recommend_reroutes(em, df, policy_clean)

    # 4. Drain rate
    df = compute_drain_rate_ledger(
        df, policy_clean,
        carrying_cost_pct=carrying_cost_pct,
        daily_storage_cost_per_unit=daily_storage_cost_per_unit,
    )

    # 5. Signal integrity (if EDI provided)
    if edi_852 is not None and not edi_852.empty:
        required_edi = ["date", "location_id", "sku_id", "reported_qty_sold"]
        if all(c in edi_852.columns for c in required_edi):
            scored_edi = score_signal_integrity(edi_852)
            summary = summarize_signal_integrity(scored_edi)
            # Join to ledger on location_id, sku_id
            df = df.merge(
                summary[["location_id", "sku_id", "signal_confidence", "confidence_tier"]],
                on=["location_id", "sku_id"],
                how="left",
            )
            df["signal_confidence"] = df["signal_confidence"].fillna(0.5)
            df["confidence_tier"] = df["confidence_tier"].fillna("MEDIUM")

    # 6. Urgency score (min days to ineligibility per lot)
    urgency_list = []
    for idx, row in df.iterrows():
        u = compute_urgency_score(row, policy_clean)
        min_days = min(u.values()) if u else 999
        urgency_list.append((idx, min_days))
    df["days_to_ineligibility"] = pd.Series({i: v for i, v in urgency_list})

    combined_result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=combined_warnings,
        cleaned_df=df,
        stats={
            "sku_rows": len(sku_clean),
            "lot_rows": len(lot_clean),
            "policy_rows": len(policy_clean),
        },
    )

    return (df, recommendations, combined_result)


def compute_at_risk_ledger(
    sku_master: pd.DataFrame,
    customer_policies: pd.DataFrame,
    lot_ledger: pd.DataFrame,
    current_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper. Runs build_comprehensive_ledger and returns ledger only.

    Use this for uploaded data in the Streamlit app or programmatic calls.
    """
    ledger, _, val_result = build_comprehensive_ledger(
        sku_master, customer_policies, lot_ledger, current_date=current_date,
    )
    if not val_result.is_valid and ledger.empty:
        raise ValueError(f"Validation failed: {[e.message for e in val_result.errors]}")
    return ledger


def main() -> None:
    """Build the At-Risk Ledger and save to data/processed/."""
    print("ReFlow AI — At-Risk Engine (Phase 2)")
    print("=" * 50)

    sku_master, customer_policies, lot_ledger = ingest_data()
    print(f"Ingested: {len(lot_ledger)} lots, {len(sku_master)} SKUs, {len(customer_policies)} customers")

    df, recommendations, val_result = build_comprehensive_ledger(
        sku_master, customer_policies, lot_ledger,
    )
    if not val_result.is_valid:
        print("Validation errors:", [e.message for e in val_result.errors])
        return
    print("Flagged at-risk lots and Financial_Risk_Exposure")
    print()

    # 5. Output
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "at_risk_ledger.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    # Save reroute recommendations
    if recommendations:
        rec_df = pd.DataFrame([
            {
                "lot_id": r.lot_id,
                "original_customer": r.original_customer,
                "recommended_customer": r.recommended_customer,
                "rsl_at_receipt": r.rsl_at_receipt,
                "rsl_headroom": r.rsl_headroom,
                "estimated_recovery_value": r.estimated_recovery_value,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r in recommendations
        ])
        rec_path = PROCESSED_DIR / "reroute_recommendations.csv"
        rec_df.to_csv(rec_path, index=False)
        print(f"Saved: {rec_path}")

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
