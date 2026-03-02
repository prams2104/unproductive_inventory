#!/usr/bin/env python3
"""
ReFlow AI — Signal Integrity Scorer

PURPOSE:
    Scores EDI 852 records for reliability using rolling z-score anomaly detection.
    Bullwhip (Lee et al. 1997): order variance exceeds sales variance. Phantom
    demand spikes (3-8x) poison planning. This scorer catches obvious spikes to
    add a "signal confidence" indicator to the ledger—preventing overreaction to
    phantom demand.

FMCG CONTEXT:
    EDI 852 = product activity/sales data. Double-counting, rationing games,
    promotion distortion create spikes. A production system would use change-point
    detection or Bayesian approaches; for MVP, z-score is sufficient.
"""
import pandas as pd


def score_signal_integrity(
    edi_852: pd.DataFrame,
    z_threshold: float = 3.0,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Score EDI 852 records for reliability using rolling z-score anomaly detection.

    For each (location_id, sku_id) pair:
    1. Sort by date, compute rolling mean and std over lookback_days
    2. z_score = (reported_qty_sold - rolling_mean) / rolling_std
    3. Flag |z_score| > z_threshold as suspect
    4. signal_confidence per location-SKU: 1.0 - (n_suspect / n_total)

    Args:
        edi_852: Columns: date, location_id, sku_id, reported_qty_sold.
        z_threshold: |z| > this → suspect.
        lookback_days: Rolling window size.

    Returns:
        DataFrame with z_score, is_suspect, signal_confidence columns added.
    """
    df = edi_852.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Ensure required columns
    required = ["date", "location_id", "sku_id", "reported_qty_sold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"EDI 852 missing columns: {missing}")

    df = df.sort_values(["location_id", "sku_id", "date"])

    # Rolling stats per (location_id, sku_id)
    df["rolling_mean"] = df.groupby(["location_id", "sku_id"])["reported_qty_sold"].transform(
        lambda x: x.rolling(lookback_days, min_periods=1).mean()
    )
    df["rolling_std"] = df.groupby(["location_id", "sku_id"])["reported_qty_sold"].transform(
        lambda x: x.rolling(lookback_days, min_periods=1).std()
    )

    # Avoid div by zero
    df["rolling_std"] = df["rolling_std"].fillna(0).replace(0, 1e-6)
    df["z_score"] = (df["reported_qty_sold"] - df["rolling_mean"]) / df["rolling_std"]
    df["is_suspect"] = df["z_score"].abs() > z_threshold

    # Per (location_id, sku_id) confidence
    n_suspect = df.groupby(["location_id", "sku_id"])["is_suspect"].transform("sum")
    n_total = df.groupby(["location_id", "sku_id"])["is_suspect"].transform("count")
    df["signal_confidence"] = 1.0 - (n_suspect / n_total)
    df["signal_confidence"] = df["signal_confidence"].clip(0.0, 1.0)

    return df.drop(columns=["rolling_mean", "rolling_std"])


def summarize_signal_integrity(
    scored_edi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate signal integrity to location-SKU level.

    Returns:
        DataFrame: location_id, sku_id, signal_confidence, n_suspect_records,
        n_total_records, confidence_tier (HIGH >0.95, MEDIUM 0.85-0.95, LOW <0.85).
    """
    agg = scored_edi.groupby(["location_id", "sku_id"]).agg(
        signal_confidence=("signal_confidence", "first"),
        n_suspect_records=("is_suspect", "sum"),
        n_total_records=("is_suspect", "count"),
    ).reset_index()

    def tier(conf: float) -> str:
        if conf > 0.95:
            return "HIGH"
        if conf >= 0.85:
            return "MEDIUM"
        return "LOW"

    agg["confidence_tier"] = agg["signal_confidence"].apply(tier)
    agg["n_suspect_records"] = agg["n_suspect_records"].astype(int)
    return agg


if __name__ == "__main__":
    """Run signal integrity on synthetic EDI 852."""
    from pathlib import Path

    print("ReFlow AI — Signal Integrity (standalone test)")
    print("=" * 55)

    project_root = Path(__file__).resolve().parent.parent
    edi = pd.read_csv(project_root / "data" / "raw_synthetic" / "edi_852_feed.csv")

    scored = score_signal_integrity(edi)
    summary = summarize_signal_integrity(scored)

    print(f"Scored {len(scored)} EDI records")
    print(f"Location-SKU pairs: {len(summary)}")
    print(f"Suspect records: {scored['is_suspect'].sum()}")
    print(f"Confidence tiers: {summary['confidence_tier'].value_counts().to_dict()}")

    print("\nDone.")
