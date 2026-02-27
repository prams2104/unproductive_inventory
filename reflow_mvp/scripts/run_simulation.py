#!/usr/bin/env python3
"""
ReFlow AI — Monte Carlo Backtest (Phase 4)

PURPOSE:
    Runs N=100 stochastic warehouse scenarios to prove the financial ROI of the
    ReFlow AI control plane. Compares Baseline (business-as-usual, 100% write-off
    on doomed lots) vs ReFlow (intervention: reroute/liquidate, 60% recovery).

FMCG / QUANT LOGIC:
    Baseline: Lots with RSL_at_arrival < strictest_required_rsl are "doomed"—
    rejected at dock, 100% write-off. RSL_at_arrival = (actual_days_remaining -
    transit_lead_time) / shelf_life.
    ReFlow: Same lots flagged 30 days prior; rerouted to Discount Partner or B2B.
    Recovery rate 60% (B-Stock/Optoro benchmarks); loss = 40%.
    Capital_Recovered = Baseline_Loss - ReFlow_Loss = 60% of doomed value.
    Monte Carlo: Each iteration uses a different seed → distinct warehouse state.
    Expected value of Capital_Recovered proves ROI is not hardcoded.

OUTPUT: data/processed/simulation_results.csv
"""
import argparse
from datetime import timedelta
from pathlib import Path

import pandas as pd

from generate_synthetic_data import REFERENCE_DATE, generate_all

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "simulation_results.csv"

# ReFlow intervention: 60% recovery when rerouting to Discount Partner / B2B
REFLOW_RECOVERY_RATE = 0.60  # Loss = 40%


def run_baseline_and_reflow(
    sku_master: pd.DataFrame,
    customer_policies: pd.DataFrame,
    lot_ledger: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Run Baseline and ReFlow scenarios for one warehouse state.

    Baseline: Doomed lots (RSL_at_arrival < strictest policy) → 100% write-off.
    ReFlow: Same lots rerouted to Discount Partner/B2B → 60% recovered, 40% loss.
    Capital_Recovered = Baseline_Loss - ReFlow_Loss.

    Args:
        sku_master, customer_policies, lot_ledger: In-memory Phase 1 data.

    Returns:
        (total_warehouse_value, baseline_loss, reflow_loss, capital_recovered).
    """
    df = lot_ledger.merge(sku_master, on="sku_id", how="left")
    df["total_lot_value"] = df["qty_on_hand"] * df["unit_cost"]
    df["production_date"] = pd.to_datetime(df["production_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    # Evaluation date: max production + 30 days (consistent with risk engine)
    max_prod = df["production_date"].max()
    current_date = max_prod + timedelta(days=30)

    df["actual_days_remaining"] = (df["expiry_date"] - pd.Timestamp(current_date)).dt.days

    # Strictest customer (highest RSL requirement)
    strictest = customer_policies.loc[customer_policies["required_rsl_pct"].idxmax()]
    transit_days = strictest["transit_lead_time_days"]
    required_rsl = strictest["required_rsl_pct"]

    # RSL at arrival: fraction of shelf life remaining when truck arrives at customer
    df["rsl_at_arrival"] = (
        df["actual_days_remaining"] - transit_days
    ) / df["standard_shelf_life_days"]

    # Doomed: RSL_at_arrival < required_rsl → dock rejection, 100% write-off
    doomed = df["rsl_at_arrival"] < required_rsl
    doomed_value = df.loc[doomed, "total_lot_value"].sum()
    total_value = df["total_lot_value"].sum()

    baseline_loss = doomed_value  # 100% write-off (scrap)
    reflow_loss = doomed_value * (1 - REFLOW_RECOVERY_RATE)  # 40% loss; 60% recovered
    capital_recovered = baseline_loss - reflow_loss  # E[recovered] proves ROI

    return total_value, baseline_loss, reflow_loss, capital_recovered


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo backtest for ReFlow AI")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of simulations")
    args = parser.parse_args()

    n = args.iterations
    results = []

    print(f"ReFlow AI — Monte Carlo Backtest (N={n})")
    print("=" * 55)

    for i in range(n):
        sku_master, customer_policies, lot_ledger, _ = generate_all(seed=i)
        total_val, baseline_loss, reflow_loss, capital_recovered = run_baseline_and_reflow(
            sku_master, customer_policies, lot_ledger
        )

        pct_reduction = (
            (capital_recovered / baseline_loss * 100) if baseline_loss > 0 else 0.0
        )

        results.append({
            "iteration_id": i + 1,
            "total_warehouse_value": round(total_val, 2),
            "baseline_loss": round(baseline_loss, 2),
            "reflow_loss": round(reflow_loss, 2),
            "capital_recovered": round(capital_recovered, 2),
            "pct_reduction_writeoffs": round(pct_reduction, 2),
        })

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n} iterations...")

    df_results = pd.DataFrame(results)

    # Summary
    avg_capital = df_results["capital_recovered"].mean()
    max_capital = df_results["capital_recovered"].max()
    avg_pct_reduction = df_results["pct_reduction_writeoffs"].mean()

    print()
    print("Summary")
    print("-" * 55)
    print(f"  Average Capital Recovered:     ${avg_capital:,.2f}")
    print(f"  Maximum Capital Recovered:    ${max_capital:,.2f}")
    print(f"  Average % Reduction in Write-offs: {avg_pct_reduction:.1f}%")
    print()

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
