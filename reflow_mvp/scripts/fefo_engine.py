#!/usr/bin/env python3
"""
ReFlow AI — FEFO Allocation Engine

PURPOSE:
    Bridges "flag it" to "route it". The risk engine flags at-risk lots; this module
    computes per-customer eligibility (RSL at receipt, headroom) and recommends
    the best alternative destination for each lot. FEFO logic: send shortest-dated
    lots to the most lenient customers that can accept them.

FMCG CONTEXT:
    A lot failing UNFI (75% RSL) may still meet Walmart (60%) or Discount Partner (10%).
    The eligibility matrix answers "where can this lot go?" The recommender ranks
    options by rsl_headroom (prefer customers where the lot has most slack) and
    recovery value. No ML, no LLM—deterministic constrained optimization.
"""
from dataclasses import dataclass

import pandas as pd


# Recovery rates by channel (industry benchmarks: B-Stock, Optoro)
RECOVERY_RATE_PRIMARY = 0.90   # Walmart, regional: 85-95%
RECOVERY_RATE_DISCOUNT = 0.50  # Discount partner: 40-60%
RECOVERY_RATE_B2B = 0.22      # B2B liquidation: 15-30%


def build_eligibility_matrix(
    enriched_ledger: pd.DataFrame,
    customer_policies: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each lot × customer pair, compute RSL at receipt, eligibility, and headroom.

    FMCG: RSL at receipt = (actual_days_remaining - transit) / shelf_life. A lot
    is eligible for customer C iff RSL_at_receipt >= C.required_rsl_pct. Headroom
    = how much slack above the minimum—drives FEFO: prefer routing to customers
    where the lot has the most remaining shelf life.

    Args:
        enriched_ledger: Lot ledger merged with SKU master; must have
            actual_days_remaining, standard_shelf_life_days, total_lot_value.
        customer_policies: required_rsl_pct, transit_lead_time_days per customer.

    Returns:
        Long DataFrame: one row per (lot_id, customer_id). Columns: lot_id,
        customer_id, customer_name, rsl_at_receipt, is_eligible, rsl_headroom.
    """
    rows = []
    for _, lot in enriched_ledger.iterrows():
        days_remaining = lot["actual_days_remaining"]
        shelf_life = lot["standard_shelf_life_days"]
        lot_id = lot["lot_id"]

        for _, cust in customer_policies.iterrows():
            transit = cust["transit_lead_time_days"]
            required_rsl = cust["required_rsl_pct"]

            # RSL when truck arrives at customer dock
            rsl_at_receipt = (days_remaining - transit) / shelf_life if shelf_life > 0 else 0.0
            is_eligible = rsl_at_receipt >= required_rsl
            rsl_headroom = rsl_at_receipt - required_rsl if is_eligible else 0.0

            rows.append({
                "lot_id": lot_id,
                "customer_id": cust["customer_id"],
                "customer_name": cust["customer_name"],
                "rsl_at_receipt": rsl_at_receipt,
                "is_eligible": is_eligible,
                "rsl_headroom": rsl_headroom,
            })

    return pd.DataFrame(rows)


@dataclass
class RerouteRecommendation:
    """
    FEFO reroute recommendation for a single at-risk lot.

    original_customer: Who it was going to fail with (strictest policy).
    recommended_customer: Best alternative destination.
    confidence: HIGH (>15% headroom), MEDIUM (5-15%), LOW (<5%).
    """
    lot_id: str
    original_customer: str
    recommended_customer: str
    rsl_at_receipt: float
    rsl_headroom: float
    estimated_recovery_value: float
    confidence: str
    reason: str


def _get_recovery_rate(customer_name: str) -> float:
    """Map customer to recovery rate by channel. FMCG: discount partners pay less."""
    name_lower = str(customer_name).lower()
    if "discount" in name_lower or "liquidat" in name_lower:
        return RECOVERY_RATE_DISCOUNT
    return RECOVERY_RATE_PRIMARY


def recommend_reroutes(
    eligibility_matrix: pd.DataFrame,
    enriched_ledger: pd.DataFrame,
    customer_policies: pd.DataFrame,
) -> list[RerouteRecommendation]:
    """
    FEFO routing logic (deterministic, no ML).

    For each at-risk lot:
    1. Get all customers where is_eligible == True
    2. Rank by: (a) rsl_headroom descending, (b) recovery_value descending
    3. If NO customer eligible → recommend B2B liquidation
    4. Assign confidence from headroom: HIGH >15%, MEDIUM 5-15%, LOW <5%

    Args:
        eligibility_matrix: Output of build_eligibility_matrix().
        enriched_ledger: Must have is_at_risk, total_lot_value.
        customer_policies: For customer names and channel classification.

    Returns:
        List of RerouteRecommendation for at-risk lots.
    """
    strictest = customer_policies.loc[customer_policies["required_rsl_pct"].idxmax()]
    strictest_name = strictest["customer_name"]
    strictest_required_rsl = float(strictest["required_rsl_pct"])
    strictest_transit = int(strictest["transit_lead_time_days"])

    at_risk_lots = enriched_ledger[enriched_ledger["is_at_risk"] == True]
    if at_risk_lots.empty:
        return []

    lot_value = dict(zip(enriched_ledger["lot_id"], enriched_ledger["total_lot_value"]))
    lot_days_remaining = dict(zip(enriched_ledger["lot_id"], enriched_ledger["actual_days_remaining"]))
    lot_shelf_life = dict(zip(enriched_ledger["lot_id"], enriched_ledger["standard_shelf_life_days"]))
    recommendations: list[RerouteRecommendation] = []

    for lot_id in at_risk_lots["lot_id"].unique():
        lot_val = lot_value.get(lot_id, 0.0)
        days_remaining = lot_days_remaining.get(lot_id, 0)
        shelf_life = lot_shelf_life.get(lot_id, 1)
        em_lot = eligibility_matrix[eligibility_matrix["lot_id"] == lot_id]

        # RSL at the strictest customer's dock — used in the rejection reason string
        rsl_at_strictest = (days_remaining - strictest_transit) / shelf_life if shelf_life > 0 else 0.0

        eligible = em_lot[em_lot["is_eligible"] == True]
        if eligible.empty:
            # No customer eligible → B2B liquidation
            recovery = lot_val * RECOVERY_RATE_B2B
            recommendations.append(RerouteRecommendation(
                lot_id=lot_id,
                original_customer=strictest_name,
                recommended_customer="B2B_Liquidation",
                rsl_at_receipt=0.0,
                rsl_headroom=0.0,
                estimated_recovery_value=recovery,
                confidence="LOW",
                reason=f"Lot {lot_id} fails all customer policies (RSL below minimum). "
                       f"Recommend B2B liquidation (B-Stock/Optoro). Estimated recovery: ${recovery:,.2f}.",
            ))
            continue

        # Rank by rsl_headroom desc, then recovery value desc
        eligible = eligible.copy()
        eligible["recovery_rate"] = eligible["customer_name"].apply(_get_recovery_rate)
        eligible["recovery_value"] = lot_val * eligible["recovery_rate"]
        eligible = eligible.sort_values(
            by=["rsl_headroom", "recovery_value"],
            ascending=[False, False],
        )

        best = eligible.iloc[0]
        rsl = best["rsl_at_receipt"]
        headroom = best["rsl_headroom"]
        rec_val = best["recovery_value"]
        rec_cust = best["customer_name"]

        if headroom >= 0.15:
            confidence = "HIGH"
        elif headroom >= 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        headroom_pct = headroom * 100
        # Use rsl_at_strictest for the rejection clause, rsl (at recommended customer) for the eligible clause
        reason = (
            f"Lot {lot_id} fails {strictest_name} "
            f"(RSL at dock {rsl_at_strictest * 100:.1f}% < {strictest_required_rsl * 100:.0f}% required). "
            f"Eligible for {rec_cust} (RSL {rsl * 100:.1f}% >= required, headroom {headroom_pct:.1f}%). "
            f"Estimated recovery: ${rec_val:,.2f}."
        )

        recommendations.append(RerouteRecommendation(
            lot_id=lot_id,
            original_customer=strictest_name,
            recommended_customer=rec_cust,
            rsl_at_receipt=rsl,
            rsl_headroom=headroom,
            estimated_recovery_value=rec_val,
            confidence=confidence,
            reason=reason,
        ))

    return recommendations


def compute_urgency_score(
    lot_row: pd.Series,
    customer_policies: pd.DataFrame,
) -> dict[str, float]:
    """
    For a given lot, compute days until ineligibility for each customer.

    FMCG: The "hard clock"—days_to_ineligibility = actual_days_remaining -
    (required_rsl_pct * shelf_life) - transit_days. Negative = already ineligible.
    Zero = today is the cliff. Drives Action Inbox priority sorting.

    Args:
        lot_row: Single row from enriched ledger (actual_days_remaining,
            standard_shelf_life_days).
        customer_policies: required_rsl_pct, transit_lead_time_days.

    Returns:
        Dict {customer_name: days_to_ineligibility}. Negative = ineligible.
    """
    days_remaining = lot_row["actual_days_remaining"]
    shelf_life = lot_row["standard_shelf_life_days"]
    if shelf_life <= 0:
        return {c["customer_name"]: -999.0 for _, c in customer_policies.iterrows()}

    result: dict[str, float] = {}
    for _, cust in customer_policies.iterrows():
        transit = cust["transit_lead_time_days"]
        required_rsl = cust["required_rsl_pct"]
        # Minimum days needed at receipt = required_rsl * shelf_life
        min_days_at_receipt = required_rsl * shelf_life
        # We need: days_remaining - transit >= min_days_at_receipt
        # So we become ineligible when: days_remaining - transit = min_days_at_receipt
        # days_to_ineligibility = (days_remaining - transit) - min_days_at_receipt
        # When positive: we have that many days until cliff
        # When zero: today is cliff
        # When negative: already past cliff
        days_to_ineligibility = (days_remaining - transit) - min_days_at_receipt
        result[cust["customer_name"]] = float(days_to_ineligibility)

    return result


if __name__ == "__main__":
    """Run FEFO engine on synthetic data."""
    from pathlib import Path

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_risk_engine import compute_at_risk_ledger, ingest_data

    print("ReFlow AI — FEFO Engine (standalone test)")
    print("=" * 55)

    sku_master, customer_policies, lot_ledger = ingest_data()
    enriched = compute_at_risk_ledger(sku_master, customer_policies, lot_ledger)

    em = build_eligibility_matrix(enriched, customer_policies)
    print(f"Eligibility matrix: {len(em)} rows (lots × customers)")

    recs = recommend_reroutes(em, enriched, customer_policies)
    print(f"Reroute recommendations: {len(recs)} for at-risk lots")

    total_recovery = sum(r.estimated_recovery_value for r in recs)
    print(f"Total estimated recovery: ${total_recovery:,.2f}")

    if recs:
        r0 = recs[0]
        print(f"\nSample: {r0.lot_id} → {r0.recommended_customer} ({r0.confidence})")
        print(f"  {r0.reason[:120]}...")

    # Urgency for first at-risk lot
    at_risk = enriched[enriched["is_at_risk"] == True]
    if not at_risk.empty:
        urgency = compute_urgency_score(at_risk.iloc[0], customer_policies)
        print(f"\nUrgency (days to ineligibility): {urgency}")

    print("\nDone.")
