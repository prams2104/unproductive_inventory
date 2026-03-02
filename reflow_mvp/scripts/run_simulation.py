#!/usr/bin/env python3
"""
ReFlow AI — Monte Carlo Backtest (Phase 4)

PURPOSE:
    Runs N=100 stochastic warehouse scenarios to prove the financial ROI of the
    ReFlow AI control plane. Uses a STOCHASTIC recovery model (not hardcoded 60%):
    recovery rate varies by channel and days-to-expiry, sampled from Beta
    distributions. Includes intervention success/failure (85% of reroutes succeed).

FMCG / QUANT LOGIC:
    Baseline: Doomed lots → 100% write-off.
    ReFlow: Reroute to best eligible customer or B2B liquidation. Recovery rate
    sampled from Beta(α, β) parameterized by channel and freshness. Days-to-expiry
    modifier: recovery degrades as expiry approaches (fire-sale territory).
    P(intervention_success)=0.85: some reroutes fail (truck unavailable, etc.).
"""
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from generate_synthetic_data import REFERENCE_DATE, generate_all

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "simulation_results.csv"

# 85% of reroute attempts succeed (truck available, customer accepts). Industry benchmark.
INTERVENTION_SUCCESS_RATE = 0.85


def sample_recovery_rate(
    channel: str,
    days_to_expiry: int,
    rng: np.random.Generator,
) -> float:
    """
    Sample a recovery rate from a Beta distribution parameterized by channel and freshness.

    Why Beta: bounded on [0, 1], flexible shape, standard for modeling rates/proportions.

    Channel parameters (B-Stock/Optoro industry benchmarks):
    - "discount_partner": Beta(α=6, β=4) → mean ~0.60, varies 0.30–0.85
    - "b2b_liquidation": Beta(α=3, β=7) → mean ~0.30, varies 0.05–0.55
    - "primary": Beta(α=8, β=2) → mean ~0.80, varies 0.50–0.98

    Days-to-expiry modifier: recovery degrades as expiry approaches.
    - days_to_expiry > 30: no penalty
    - 15 < days <= 30: multiply α by 0.85
    - 7 < days <= 15: multiply α by 0.65
    - days <= 7: multiply α by 0.40 (fire-sale)
    """
    channel_lower = channel.lower()
    if "b2b" in channel_lower or "liquidation" in channel_lower:
        alpha, beta = 3.0, 7.0
    elif "discount" in channel_lower:
        alpha, beta = 6.0, 4.0
    else:
        alpha, beta = 8.0, 2.0

    # Freshness penalty
    if days_to_expiry <= 7:
        alpha *= 0.40
    elif days_to_expiry <= 15:
        alpha *= 0.65
    elif days_to_expiry <= 30:
        alpha *= 0.85
    # else: no penalty

    alpha = max(0.1, alpha)
    return float(rng.beta(alpha, beta))


def run_baseline_and_reflow(
    sku_master: pd.DataFrame,
    customer_policies: pd.DataFrame,
    lot_ledger: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """
    Run Baseline and ReFlow scenarios for one warehouse state.

    ReFlow: Uses FEFO engine to get reroute recommendations. For each at-risk lot,
    samples recovery rate by channel and days-to-expiry. Applies intervention
    success (85% succeed, 15% fail → 0 recovery).

    Returns:
        Dict with iteration metrics including recovery_rate_mean, recovery_rate_std.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_risk_engine import compute_at_risk_ledger
    from fefo_engine import build_eligibility_matrix, recommend_reroutes

    df = lot_ledger.merge(sku_master, on="sku_id", how="left")
    df["total_lot_value"] = df["qty_on_hand"] * df["unit_cost"]
    df["production_date"] = pd.to_datetime(df["production_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    max_prod = df["production_date"].max()
    current_date = max_prod + timedelta(days=30)
    df["actual_days_remaining"] = (df["expiry_date"] - pd.Timestamp(current_date)).dt.days

    strictest = customer_policies.loc[customer_policies["required_rsl_pct"].idxmax()]
    transit_days = strictest["transit_lead_time_days"]
    required_rsl = strictest["required_rsl_pct"]

    df["rsl_at_arrival"] = (
        df["actual_days_remaining"] - transit_days
    ) / df["standard_shelf_life_days"]

    doomed = df["rsl_at_arrival"] < required_rsl
    doomed_value = df.loc[doomed, "total_lot_value"].sum()
    total_value = df["total_lot_value"].sum()
    baseline_loss = doomed_value

    # ReFlow: run risk engine + FEFO to get recommendations
    enriched = compute_at_risk_ledger(sku_master, customer_policies, lot_ledger)
    em = build_eligibility_matrix(enriched, customer_policies)
    recs = recommend_reroutes(em, enriched, customer_policies)

    # Build lot_id -> (channel, days_to_expiry, value) for doomed lots
    doomed_lot_ids = set(df.loc[doomed, "lot_id"])
    lot_days = dict(zip(df["lot_id"], df["actual_days_remaining"]))
    lot_value = dict(zip(df["lot_id"], df["total_lot_value"]))

    recovery_rates: list[float] = []
    capital_recovered = 0.0
    n_rerouted = 0
    n_reroute_failed = 0
    n_liquidated = 0

    for rec in recs:
        if rec.lot_id not in doomed_lot_ids:
            continue
        val = lot_value.get(rec.lot_id, 0.0)
        days = int(lot_days.get(rec.lot_id, 30))

        channel = rec.recommended_customer
        success = rng.random() < INTERVENTION_SUCCESS_RATE

        if success:
            rate = sample_recovery_rate(channel, days, rng)
            recovery_rates.append(rate)
            capital_recovered += val * rate
            if "B2B" in channel or "Liquidation" in channel:
                n_liquidated += 1
            else:
                n_rerouted += 1
        else:
            n_reroute_failed += 1

    reflow_loss = doomed_value - capital_recovered

    return {
        "total_warehouse_value": total_value,
        "n_at_risk_lots": len(doomed_lot_ids),
        "n_rerouted": n_rerouted,
        "n_reroute_failed": n_reroute_failed,
        "n_liquidated": n_liquidated,
        "baseline_loss": baseline_loss,
        "reflow_loss": reflow_loss,
        "capital_recovered": capital_recovered,
        "recovery_rate_mean": np.mean(recovery_rates) if recovery_rates else 0.0,
        "recovery_rate_std": np.std(recovery_rates) if len(recovery_rates) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo backtest for ReFlow AI")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of simulations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    n = args.iterations
    rng = np.random.default_rng(args.seed)
    results = []

    print(f"ReFlow AI — Monte Carlo Backtest (N={n}, seed={args.seed})")
    print("=" * 55)

    for i in range(n):
        sku_master, customer_policies, lot_ledger, _ = generate_all(seed=i)
        iter_rng = np.random.default_rng(args.seed + i)
        row = run_baseline_and_reflow(sku_master, customer_policies, lot_ledger, iter_rng)

        pct_reduction = (
            (row["capital_recovered"] / row["baseline_loss"] * 100)
            if row["baseline_loss"] > 0 else 0.0
        )

        results.append({
            "iteration_id": i + 1,
            "total_warehouse_value": round(row["total_warehouse_value"], 2),
            "n_at_risk_lots": row["n_at_risk_lots"],
            "n_rerouted": row["n_rerouted"],
            "n_reroute_failed": row["n_reroute_failed"],
            "n_liquidated": row["n_liquidated"],
            "baseline_loss": round(row["baseline_loss"], 2),
            "reflow_loss": round(row["reflow_loss"], 2),
            "capital_recovered": round(row["capital_recovered"], 2),
            "pct_reduction_writeoffs": round(pct_reduction, 2),
            "recovery_rate_mean": round(row["recovery_rate_mean"], 4),
            "recovery_rate_std": round(row["recovery_rate_std"], 4),
        })

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n} iterations...")

    df_results = pd.DataFrame(results)

    avg_capital = df_results["capital_recovered"].mean()
    max_capital = df_results["capital_recovered"].max()
    avg_pct_reduction = df_results["pct_reduction_writeoffs"].mean()
    avg_recovery_rate = df_results["recovery_rate_mean"].mean()

    print()
    print("Summary")
    print("-" * 55)
    print(f"  Average Capital Recovered:     ${avg_capital:,.2f}")
    print(f"  Maximum Capital Recovered:     ${max_capital:,.2f}")
    print(f"  Average % Reduction in Write-offs: {avg_pct_reduction:.1f}%")
    print(f"  Mean Recovery Rate (across iterations): {avg_recovery_rate*100:.1f}%")
    print()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
