#!/usr/bin/env python3
"""
ReFlow AI — Synthetic FMCG Dataset Generator (Phase 1: Thin Data Contract)

PURPOSE:
    Generates a bias-free, stochastic FMCG dataset for the ReFlow AI MVP. Unlike
    hardcoded demos, at-risk lots emerge organically from probability distributions—
    removing confirmation bias and enabling valid Monte Carlo validation.

FMCG / QUANT LOGIC:
    - Production dates: N(μ, σ²) on age_ratio (fraction of shelf life consumed).
      Most lots are fresh; the right tail (age_ratio > 0.25) yields at-risk lots
      naturally. Tuned to hit 1.5–3% failure rate (FMCG industry benchmark).
    - EDI 852 demand: Poisson(λ)—the canonical distribution for count data
      (units sold per day). Discrete, non-negative, variance = mean. Normal would
      allow negative demand.
    - Transit lead time: base + N(0, 1)—exogenous supply chain chaos. A truck
      arriving one day late can push a marginal lot below the retailer's RSL
      threshold, causing dock rejection.

OUTPUT:
    data/raw_synthetic/: sku_master.csv, customer_policies.csv, lot_ledger.csv,
    edi_852_feed.csv

DEPENDENCIES:
    pandas, numpy, faker (optional)
"""
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_synthetic"

# Evaluation anchor: "today" for RSL calculations at generation time.
# NOTE: compute_organic_failure_rate() measures the failure rate AT this date (~1.5–3%).
# build_risk_engine.compute_current_date() evaluates at max(production_date) + 30 days,
# which is ~30 days AFTER REFERENCE_DATE. This intentional offset simulates "the next
# replenishment review cycle" and will flag significantly more lots as at-risk (typically
# 60–80%) — that's the operational reality, not a calibration error.
REFERENCE_DATE = datetime(2025, 2, 26)

# FMCG benchmark: organic failure rate (at-risk + expired) as % of inventory value
TARGET_FAILURE_RATE_PCT = (0.015, 0.03)  # 1.5% to 3%


def set_seed(seed: int | None) -> None:
    """
    Set all RNG seeds for reproducibility.

    Args:
        seed: Integer seed for numpy, random, and Faker. None = non-deterministic.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        if HAS_FAKER:
            Faker.seed(seed)


def ensure_project_structure() -> None:
    """Create project directory structure before generating data."""
    dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "raw_synthetic",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "scripts",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SKU MASTER (unchanged structure; uses RNG)
# ---------------------------------------------------------------------------
def generate_sku_master(n_skus: int = 60) -> pd.DataFrame:
    """
    Generate SKU master with category-appropriate shelf lives.

    FMCG logic: Retailers enforce receiving minimums by category. Dairy (perishable)
    has short shelf life; Beverage (shelf-stable) has long. Unit cost scales with
    perishability.

    Args:
        n_skus: Number of SKUs to generate.

    Returns:
        DataFrame with columns: sku_id, category, unit_cost, standard_shelf_life_days.
    """
    categories = ["Dairy", "Snacks", "Beverage"]
    weights = [0.25, 0.40, 0.35]
    rows = []
    for i in range(n_skus):
        cat = np.random.choice(categories, p=weights)
        if cat == "Dairy":
            shelf_life = int(np.random.uniform(30, 61))
            unit_cost = round(np.random.uniform(2.0, 12.0), 2)
        elif cat == "Snacks":
            shelf_life = int(np.random.uniform(90, 181))
            unit_cost = round(np.random.uniform(0.5, 5.0), 2)
        else:
            shelf_life = int(np.random.uniform(180, 366))
            unit_cost = round(np.random.uniform(0.3, 4.0), 2)
        rows.append({
            "sku_id": f"SKU-{i+1:04d}",
            "category": cat,
            "unit_cost": unit_cost,
            "standard_shelf_life_days": shelf_life,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. CUSTOMER POLICIES (transit lead time with exogenous noise)
# ---------------------------------------------------------------------------
def generate_customer_policies(seed: int | None = None) -> pd.DataFrame:
    """
    Generate customer shelf-life policies with stochastic transit lead times.

    FMCG logic: UNFI requires 75% RSL at receipt; Walmart ~60%; Discount Partner
    (liquidator) accepts 10%. Transit_Lead_Time = base + N(0, 1) simulates supply
    chain chaos—a day late can push a marginal lot below the threshold.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns: customer_id, customer_name, required_rsl_pct,
        transit_lead_time_days.
    """
    set_seed(seed)
    if HAS_FAKER:
        fake = Faker()
        extra_names = [fake.company(), fake.company()]
    else:
        extra_names = ["Regional_Retailer_A", "Regional_Retailer_B"]
    base_policies = [
        ("CUST-001", "UNFI", 0.75, 5),
        ("CUST-002", "Walmart", 0.60, 7),
        ("CUST-003", "Discount_Partner", 0.10, 3),
        ("CUST-004", extra_names[0], 0.65, 6),
        ("CUST-005", extra_names[1], 0.70, 5),
    ]
    rows = []
    for cid, name, rsl, base_days in base_policies:
        # Exogenous noise: base + N(0, 1), rounded, floored at 1
        noise = np.random.normal(0, 1)
        transit_days = max(1, int(round(base_days + noise)))
        rows.append({
            "customer_id": cid,
            "customer_name": name,
            "required_rsl_pct": rsl,
            "transit_lead_time_days": transit_days,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. LOT LEDGER (stochastic production dates)
# ---------------------------------------------------------------------------
def generate_lot_ledger(
    sku_master: pd.DataFrame,
    reference_date: datetime,
    n_lots: int = 200,
    n_locations: int = 5,
    age_mean_pct: float = 0.10,
    age_std_pct: float = 0.08,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate lot ledger with stochastic production dates (no hardcoded at-risk).

    FMCG / QUANT LOGIC:
        RSL = (expiry - today) / (expiry - production) = remaining_days / shelf_life.
        For UNFI (75% min), at-risk means RSL < 0.75, i.e. age_ratio > 0.25.
        We use age_ratio ~ N(0.10, 0.08): mean 10% consumed, std 8%. The right tail
        (age_ratio > 0.25) has ~3% probability—matching FMCG benchmark. No forced
        failures; they emerge from the Gaussian tail.

    Args:
        sku_master: SKU catalog with standard_shelf_life_days.
        reference_date: "Today" for RSL evaluation.
        n_lots, n_locations: Warehouse scale.
        age_mean_pct: Mean fraction of shelf life consumed (replenishment cycle).
        age_std_pct: Std dev; tunes organic failure rate.

    Returns:
        DataFrame with columns: lot_id, sku_id, location_id, qty_on_hand,
        production_date, expiry_date.
    """
    set_seed(seed)
    sku_ids = sku_master["sku_id"].tolist()
    sku_shelf_life = dict(zip(sku_master["sku_id"], sku_master["standard_shelf_life_days"]))
    location_ids = [f"LOC-{i+1:02d}" for i in range(n_locations)]

    rows = []
    for i in range(n_lots):
        sku_id = np.random.choice(sku_ids)
        shelf_life = sku_shelf_life[sku_id]
        location_id = np.random.choice(location_ids)

        # Age (fraction of shelf life consumed) ~ N(μ, σ²)
        # Clip to [0.01, 1.2] to allow some expired lots (age_ratio > 1)
        age_ratio = np.clip(np.random.normal(age_mean_pct, age_std_pct), 0.01, 1.2)
        age_days = int(age_ratio * shelf_life)
        age_days = max(1, min(age_days, int(shelf_life * 1.1)))  # Keep sensible range

        production_date = reference_date - timedelta(days=age_days)
        expiry_date = production_date + timedelta(days=shelf_life)
        qty_on_hand = int(np.random.uniform(50, 2001))

        rows.append({
            "lot_id": f"LOT-{i+1:05d}",
            "sku_id": sku_id,
            "location_id": location_id,
            "qty_on_hand": qty_on_hand,
            "production_date": production_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. EDI 852 FEED (Poisson demand + bullwhip spikes)
# ---------------------------------------------------------------------------
def generate_edi_852_feed(
    sku_master: pd.DataFrame,
    lot_ledger: pd.DataFrame,
    reference_date: datetime,
    n_days: int = 90,
    spike_probability: float = 0.08,
    spike_multiplier_range: tuple[float, float] = (3.0, 8.0),
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate EDI 852 product activity with Poisson demand and bullwhip spikes.

    FMCG / QUANT LOGIC:
        Poisson(λ): Canonical for count data (units sold/day). Discrete, non-negative,
        variance = mean. Normal would allow negative demand. λ varies by category
        (Dairy < Snacks < Beverage) and day-of-week (dow_factor).
        Bullwhip: Lee et al. (1997)—order variance exceeds sales variance. We inject
        phantom spikes (3–8x) on ~8% of records to simulate double-counting,
        rationing games, promotion distortion. ReFlow's Signal Integrity Scorer
        must detect these.

    Args:
        sku_master, lot_ledger: Reference data.
        reference_date: Anchor for date range.
        n_days: Days of activity.
        spike_probability: P(phantom spike).
        spike_multiplier_range: Multiplier range for spikes.

    Returns:
        DataFrame with columns: date, location_id, sku_id, reported_qty_sold.
    """
    set_seed(seed)
    location_ids = lot_ledger["location_id"].unique().tolist()
    sku_ids = sku_master["sku_id"].tolist()
    rng = np.random.default_rng(seed)

    location_sku_pairs = [
        (loc, sku)
        for loc in location_ids
        for sku in list(np.random.choice(sku_ids, size=min(40, len(sku_ids)), replace=False))
    ]

    base_lam = {"Dairy": 80.0, "Snacks": 120.0, "Beverage": 150.0}
    rows = []
    for loc, sku in location_sku_pairs:
        cat = sku_master[sku_master["sku_id"] == sku].iloc[0]["category"]
        lam_base = base_lam[cat]
        for d in range(n_days):
            date = reference_date - timedelta(days=n_days - d)
            dow_factor = 1.0 + 0.2 * np.sin(d / 7)  # Weekly seasonality
            lam = max(0.1, lam_base * dow_factor)
            demand = int(rng.poisson(lam))  # Poisson: count data, variance=mean
            if np.random.random() < spike_probability:
                mult = np.random.uniform(*spike_multiplier_range)
                reported_qty = min(int(demand * mult), 5000)
            else:
                reported_qty = demand
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "location_id": loc,
                "sku_id": sku,
                "reported_qty_sold": reported_qty,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ORCHESTRATION (Monte Carlo–ready)
# ---------------------------------------------------------------------------
def generate_all(
    seed: int | None = 42,
    reference_date: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate full dataset in memory (no CSV writes). Monte Carlo–ready.

    Args:
        seed: RNG seed. Use different seeds (0..999) for distinct warehouse states.
        reference_date: Evaluation anchor. Default REFERENCE_DATE.

    Returns:
        (sku_master, customer_policies, lot_ledger, edi_852) as DataFrames.
    """
    ref = reference_date or REFERENCE_DATE
    set_seed(seed)

    sku_master = generate_sku_master()
    customer_policies = generate_customer_policies(seed=seed)
    lot_ledger = generate_lot_ledger(sku_master, ref, seed=seed)
    edi_852 = generate_edi_852_feed(sku_master, lot_ledger, ref, seed=seed)

    # Assert synthetic data passes schema validation (pilot readiness)
    from schema import validate_dataframe, validate_referential_integrity
    assert validate_dataframe(sku_master, "sku_master").is_valid, "SKU Master must pass validation"
    assert validate_dataframe(customer_policies, "customer_policies").is_valid, "Customer Policies must pass validation"
    lot_res = validate_dataframe(lot_ledger, "lot_ledger")
    assert lot_res.is_valid and lot_res.cleaned_df is not None, "Lot Ledger must pass validation"
    ref_res = validate_referential_integrity(lot_res.cleaned_df, sku_master)
    assert ref_res.cleaned_df is not None and len(ref_res.cleaned_df) > 0, "Referential integrity must hold"

    return sku_master, customer_policies, lot_ledger, edi_852


def compute_organic_failure_rate(
    lot_ledger: pd.DataFrame,
    sku_master: pd.DataFrame,
    reference_date: datetime,
    strictest_rsl: float = 0.75,
) -> tuple[float, int, float]:
    """
    Compute organic failure rate (no forced at-risk). Validates benchmark calibration.

    RSL = actual_days_remaining / shelf_life. At-risk = RSL < strictest_rsl and
    not expired. Expired = actual_days_remaining <= 0.

    Args:
        lot_ledger, sku_master: Merged to get total_lot_value, shelf_life.
        reference_date: "Today" for RSL.
        strictest_rsl: Threshold (e.g., 0.75 for UNFI).

    Returns:
        (failure_rate_pct, n_at_risk, total_value).
    """
    df = lot_ledger.merge(sku_master, on="sku_id", how="left")
    df["total_lot_value"] = df["qty_on_hand"] * df["unit_cost"]
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["production_date"] = pd.to_datetime(df["production_date"])
    df["actual_days_remaining"] = (df["expiry_date"] - pd.Timestamp(reference_date)).dt.days
    # Use standard_shelf_life_days from SKU master — matches production path in build_risk_engine.py.
    # Dividing by (expiry - production).days would diverge for lots with manual expiry overrides.
    df["rsl"] = df["actual_days_remaining"] / df["standard_shelf_life_days"]
    at_risk = (df["rsl"] < strictest_rsl) & (df["actual_days_remaining"] > 0)
    expired = df["actual_days_remaining"] <= 0
    failed_value = df.loc[at_risk | expired, "total_lot_value"].sum()
    total_value = df["total_lot_value"].sum()
    rate = (failed_value / total_value * 100) if total_value > 0 else 0.0
    return rate, int(at_risk.sum()), total_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stochastic FMCG synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (use different for Monte Carlo)")
    parser.add_argument("--no-save", action="store_true", help="Print stats only, do not write CSVs")
    args = parser.parse_args()

    ensure_project_structure()
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sku_master, customer_policies, lot_ledger, edi_852 = generate_all(seed=args.seed)

    # Report organic failure rate (no forced at-risk)
    failure_rate, n_at_risk, _ = compute_organic_failure_rate(
        lot_ledger, sku_master, REFERENCE_DATE
    )
    failure_rate_decimal = failure_rate / 100.0
    in_benchmark = TARGET_FAILURE_RATE_PCT[0] <= failure_rate_decimal <= TARGET_FAILURE_RATE_PCT[1]

    print(f"ReFlow AI — Stochastic Data Generator (seed={args.seed})")
    print("=" * 55)
    print(f"  sku_master.csv: {len(sku_master)} SKUs")
    print(f"  customer_policies.csv: {len(customer_policies)} customers")
    print(f"  lot_ledger.csv: {len(lot_ledger)} lots")
    print(f"  edi_852_feed.csv: {len(edi_852)} records")
    print()
    print("Organic failure rate (no forced at-risk):")
    print(f"  Lots at-risk (RSL < 75%): {n_at_risk}")
    print(f"  Failure rate: {failure_rate:.2f}% of inventory value")
    print(f"  Target benchmark: {TARGET_FAILURE_RATE_PCT[0]*100:.1f}–{TARGET_FAILURE_RATE_PCT[1]*100:.1f}%")
    print(f"  In benchmark: {'Yes' if in_benchmark else 'No'}")
    print()

    if not args.no_save:
        sku_master.to_csv(RAW_DATA_DIR / "sku_master.csv", index=False)
        customer_policies.to_csv(RAW_DATA_DIR / "customer_policies.csv", index=False)
        lot_ledger.to_csv(RAW_DATA_DIR / "lot_ledger.csv", index=False)
        edi_852.to_csv(RAW_DATA_DIR / "edi_852_feed.csv", index=False)
        print(f"Saved to {RAW_DATA_DIR}/")
    print("Done. Monte Carlo: python scripts/generate_synthetic_data.py --seed <n>")


if __name__ == "__main__":
    main()
