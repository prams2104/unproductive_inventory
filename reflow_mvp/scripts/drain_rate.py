#!/usr/bin/env python3
"""
ReFlow AI — Drain Rate Calculator

PURPOSE:
    Computes the daily economic drain for each at-risk lot. Financial risk exposure
    is binary (full value or 0) in the base engine; drain rate captures the reality
    that a lot with 45 days remaining is losing value EVERY DAY it sits—not just
    on the ineligibility cliff. Drives Action Inbox priority: highest-drain lots first.

FMCG CONTEXT:
    Carrying cost = capital cost + storage cost. Industry benchmark ~25% annual.
    Risk multiplier: as days-to-ineligibility shrink, the cost of inaction escalates
    (elevated → critical → emergency). This is the number for the CFO board slide.
"""
from typing import Any

import pandas as pd


def compute_drain_rate(
    lot_row: pd.Series,
    carrying_cost_pct: float = 0.25,
    daily_storage_cost_per_unit: float = 0.02,
    days_to_ineligibility: float | None = None,
) -> dict[str, Any]:
    """
    Compute the daily economic drain for a single lot.

    Components (standard carrying cost breakdown):
    1. Capital cost: (unit_cost * qty * carrying_cost_pct) / 365
    2. Storage cost: qty * daily_storage_cost_per_unit
    3. Risk cost: value_decay_rate from proximity to ineligibility cliff
       - days > 30: risk_multiplier = 1.0
       - 15 < days <= 30: 1.5
       - 7 < days <= 15: 2.5
       - days <= 7: 5.0 (emergency)

    drain_rate_daily = (capital_cost_daily + storage_cost_daily) * risk_multiplier

    Also: cumulative_drain, projected_total_drain, break_even_days.

    Args:
        lot_row: Must have qty_on_hand, unit_cost, total_lot_value (or qty*unit_cost),
            production_date, expiry_date. Optional: days_to_ineligibility (if precomputed).
        carrying_cost_pct: Annual carrying cost (opportunity cost of capital).
        daily_storage_cost_per_unit: $ per unit per day.

    Returns:
        Dict with drain_rate_daily, risk_multiplier, cumulative_drain,
        projected_total_drain, break_even_days, etc.
    """
    qty = lot_row.get("qty_on_hand", 0)
    unit_cost = lot_row.get("unit_cost", 0.0)
    total_value = lot_row.get("total_lot_value", qty * unit_cost)
    production = pd.Timestamp(lot_row.get("production_date"))
    expiry = pd.Timestamp(lot_row.get("expiry_date"))
    today = pd.Timestamp("today").normalize()

    if days_to_ineligibility is None:
        # Use days to expiry as proxy if not provided
        days_to_ineligibility = (expiry - today).days

    # Risk multiplier from days to ineligibility
    if days_to_ineligibility <= 7:
        risk_multiplier = 5.0
    elif days_to_ineligibility <= 15:
        risk_multiplier = 2.5
    elif days_to_ineligibility <= 30:
        risk_multiplier = 1.5
    else:
        risk_multiplier = 1.0

    capital_cost_daily = (total_value * carrying_cost_pct) / 365
    storage_cost_daily = qty * daily_storage_cost_per_unit
    drain_rate_daily = (capital_cost_daily + storage_cost_daily) * risk_multiplier

    days_already_held = (today - production).days
    days_to_expiry = max(0, (expiry - today).days)
    cumulative_drain = drain_rate_daily * max(0, days_already_held)
    projected_total_drain = drain_rate_daily * days_to_expiry if days_to_expiry > 0 else 0.0

    # Break-even: days until cumulative drain exceeds salvage (assume 30% salvage)
    salvage_value = total_value * 0.30
    break_even_drain = salvage_value
    if drain_rate_daily > 0:
        break_even_days = break_even_drain / drain_rate_daily
    else:
        break_even_days = float("inf")

    return {
        "drain_rate_daily": drain_rate_daily,
        "risk_multiplier": risk_multiplier,
        "capital_cost_daily": capital_cost_daily,
        "storage_cost_daily": storage_cost_daily,
        "cumulative_drain": cumulative_drain,
        "projected_total_drain": projected_total_drain,
        "days_to_expiry": days_to_expiry,
        "break_even_days": break_even_days,
    }


def compute_drain_rate_ledger(
    enriched_ledger: pd.DataFrame,
    customer_policies: pd.DataFrame,
    carrying_cost_pct: float = 0.25,
    daily_storage_cost_per_unit: float = 0.02,
    days_to_ineligibility_col: str | None = None,
) -> pd.DataFrame:
    """
    Apply drain rate computation to entire ledger.

    For at-risk lots, uses minimum days_to_ineligibility across customers (strictest
    cliff). Sorts by drain_rate_daily descending—highest-drain lots need action first.

    Args:
        enriched_ledger: Must have is_at_risk, qty_on_hand, unit_cost, total_lot_value,
            production_date, expiry_date.
        customer_policies: For computing days-to-ineligibility if not provided.
        days_to_ineligibility_col: If ledger has precomputed min days to cliff.

    Returns:
        Ledger with drain columns added, sorted by drain_rate_daily desc.
    """
    from fefo_engine import compute_urgency_score

    df = enriched_ledger.copy()
    drain_cols = [
        "drain_rate_daily", "risk_multiplier", "capital_cost_daily", "storage_cost_daily",
        "cumulative_drain", "projected_total_drain", "days_to_expiry", "break_even_days",
    ]
    # Skip if already computed (e.g. from build_comprehensive_ledger)
    if all(c in df.columns for c in drain_cols):
        return df.sort_values("drain_rate_daily", ascending=False)

    new_cols = {c: [] for c in drain_cols}
    for _, row in df.iterrows():
        days_to_inelig = None
        if days_to_ineligibility_col and days_to_ineligibility_col in row:
            days_to_inelig = row[days_to_ineligibility_col]
        elif row.get("is_at_risk", False):
            urgency = compute_urgency_score(row, customer_policies)
            days_to_inelig = min(urgency.values()) if urgency else 30
        else:
            expiry = pd.Timestamp(row.get("expiry_date"))
            days_to_inelig = (expiry - pd.Timestamp("today").normalize()).days

        drain = compute_drain_rate(
            row,
            carrying_cost_pct=carrying_cost_pct,
            daily_storage_cost_per_unit=daily_storage_cost_per_unit,
            days_to_ineligibility=days_to_inelig,
        )
        for c in drain_cols:
            new_cols[c].append(drain.get(c, 0.0))

    for c in drain_cols:
        df[c] = new_cols[c]

    df = df.sort_values("drain_rate_daily", ascending=False)
    return df


if __name__ == "__main__":
    """Run drain rate on synthetic data."""
    from pathlib import Path

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_risk_engine import compute_at_risk_ledger, ingest_data

    print("ReFlow AI — Drain Rate (standalone test)")
    print("=" * 55)

    sku_master, customer_policies, lot_ledger = ingest_data()
    enriched = compute_at_risk_ledger(sku_master, customer_policies, lot_ledger)

    drained = compute_drain_rate_ledger(enriched, customer_policies)
    at_risk = drained[drained["is_at_risk"] == True]

    if not at_risk.empty:
        total_daily_drain = at_risk["drain_rate_daily"].sum()
        print(f"Total daily capital drain (at-risk lots): ${total_daily_drain:,.2f}")
        print(f"Top 3 by drain: {at_risk[['lot_id', 'drain_rate_daily']].head(3).to_string()}")

    print("\nDone.")
