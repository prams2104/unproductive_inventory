"""
ReFlow AI | Action Inbox (Phase 3)

PURPOSE:
    Governed Action Inbox UI for supply chain planners to review at-risk inventory
    and approve/deny agent-recommended actions. Human-in-the-loop for NIST AI RMF
    compliance—no autonomous execution without approval.

FEATURES:
    - Tab 1 (Live Action Inbox): At-risk lot grid, KPI metrics, action simulator.
    - Tab 2 (ROI Validation): Monte Carlo backtest results.
    - Tab 3 (Run on Your Data): Upload CSV exports and run the risk engine.
    - Audit trail: All actions logged to data/processed/audit_log.csv.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from build_risk_engine import build_comprehensive_ledger, compute_at_risk_ledger

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ReFlow AI | Action Inbox",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------------------------------------------------------------------
# PATHS & DATA LOADING
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
LEDGER_PATH = PROJECT_ROOT / "data" / "processed" / "at_risk_ledger.csv"
RECOMMENDATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "reroute_recommendations.csv"
AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "audit_log.csv"
SIMULATION_PATH = PROJECT_ROOT / "data" / "processed" / "simulation_results.csv"


@st.cache_data
def load_at_risk_ledger():
    """
    Load at-risk ledger (Phase 2 output). Filters to is_at_risk == True.

    Returns:
        DataFrame or None if file not found.
    """
    if not LEDGER_PATH.exists():
        return None
    df = pd.read_csv(LEDGER_PATH)
    df = df[df["is_at_risk"] == True].copy()
    return df


@st.cache_data
def load_reroute_recommendations():
    """Load FEFO reroute recommendations. Returns DataFrame or None."""
    if not RECOMMENDATIONS_PATH.exists():
        return None
    return pd.read_csv(RECOMMENDATIONS_PATH)


@st.cache_data
def load_simulation_results():
    """
    Load Monte Carlo simulation results (Phase 4 output).

    Returns:
        DataFrame or None if file not found.
    """
    if not SIMULATION_PATH.exists():
        return None
    import pandas as pd
    return pd.read_csv(SIMULATION_PATH)


def append_audit_log(lot_id: str, action: str, user: str = "Planner") -> None:
    """
    Append action to audit log. NIST AI RMF: every decision must be traceable.

    Args:
        lot_id, action, user: Logged to data/processed/audit_log.csv.
    """
    import pandas as pd
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lot_id": lot_id,
        "action": action,
        "user": user,
    }
    new_row = pd.DataFrame([row])
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not AUDIT_LOG_PATH.exists()
    new_row.to_csv(AUDIT_LOG_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# TAB 1: LIVE ACTION INBOX
# ---------------------------------------------------------------------------
def _signal_indicator(tier: str) -> str:
    """Return colored dot for signal confidence tier."""
    if tier == "HIGH":
        return "🟢"
    if tier == "MEDIUM":
        return "🟡"
    return "🔴"


def render_action_inbox_tab():
    """
    Render Live Action Inbox: KPIs, at-risk lot grid (sorted by drain), action simulator.
    Shows FEFO reroute recommendation per lot; Approve Recommended Action button.
    """
    df = load_at_risk_ledger()
    if df is None:
        st.error(
            f"At-risk ledger not found at `{LEDGER_PATH}`. "
            "Run `python scripts/build_risk_engine.py` first."
        )
        return
    if df.empty:
        st.success("No lots currently at risk. All inventory meets customer shelf-life policies.")
        st.balloons()
        return

    recs_df = load_reroute_recommendations()
    rec_by_lot = {}
    if recs_df is not None and not recs_df.empty:
        for _, r in recs_df.iterrows():
            rec_by_lot[r["lot_id"]] = r

    if "acted_lots" not in st.session_state:
        st.session_state.acted_lots = set()

    df_remaining = df[~df["lot_id"].isin(st.session_state.acted_lots)]
    # Sort by drain rate (highest economic urgency first)
    if "drain_rate_daily" in df_remaining.columns:
        df_remaining = df_remaining.sort_values("drain_rate_daily", ascending=False)
    total_lots = len(df_remaining)
    total_exposure = df_remaining["financial_risk_exposure"].sum()
    total_drain = df_remaining["drain_rate_daily"].sum() if "drain_rate_daily" in df_remaining.columns else 0
    total_recoverable = sum(
        rec_by_lot.get(lid, {}).get("estimated_recovery_value", 0) or 0
        for lid in df_remaining["lot_id"]
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Lots at Risk", f"{total_lots:,}")
    with col2:
        st.metric("Total Financial Risk Exposure", f"${total_exposure:,.2f}")
    with col3:
        st.metric("Total Daily Capital Drain", f"${total_drain:,.2f}")
    with col4:
        st.metric("Total Recoverable Value", f"${total_recoverable:,.2f}")

    st.divider()
    st.subheader("At-Risk Lot Ledger (sorted by economic urgency)")
    display_cols = ["lot_id", "sku_id", "location_id", "total_lot_value", "actual_days_remaining"]
    if "drain_rate_daily" in df_remaining.columns:
        display_cols.extend(["drain_rate_daily", "projected_total_drain"])
    if "confidence_tier" in df_remaining.columns:
        display_cols.append("confidence_tier")
    display_cols = [c for c in display_cols if c in df_remaining.columns]
    df_display = df_remaining[display_cols].copy()
    if "confidence_tier" in df_display.columns:
        df_display["Signal"] = df_display["confidence_tier"].apply(_signal_indicator)
    st.dataframe(
        df_display.rename(columns={
            "lot_id": "Lot ID",
            "sku_id": "SKU ID",
            "location_id": "Location ID",
            "total_lot_value": "Total Lot Value",
            "actual_days_remaining": "Days Left",
            "drain_rate_daily": "Daily Drain ($)",
            "projected_total_drain": "Projected Drain ($)",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Action Simulator")
    available_lots = df_remaining
    if available_lots.empty:
        st.success("All at-risk lots have been acted on.")
        st.balloons()
        return

    lot_options = available_lots["lot_id"].tolist()
    selected_lot = st.selectbox(
        "Select a lot to act on",
        options=lot_options,
        index=0,
        key="lot_selector",
    )

    if selected_lot:
        lot_row = df[df["lot_id"] == selected_lot].iloc[0]
        rec = rec_by_lot.get(selected_lot)

        if rec is not None:
            st.info(
                f"**FEFO Recommendation:** {rec['recommended_customer']} | "
                f"Recovery: ${rec['estimated_recovery_value']:,.2f} | Confidence: {rec['confidence']}"
            )
            st.caption(rec.get("reason", "")[:200] + ("..." if len(str(rec.get("reason", ""))) > 200 else ""))

        col1, col2, col3 = st.columns(3)
        with col1:
            if rec is not None and rec.get("recommended_customer") not in ("B2B_Liquidation",):
                reroute_clicked = st.button(
                    "✅ Approve Recommended Action",
                    use_container_width=True,
                    help=f"Reroute to {rec.get('recommended_customer', 'recommended customer')}",
                )
            else:
                st.caption("No reroute recommendation (B2B liquidation only)")
                reroute_clicked = False
        with col2:
            liquidate_clicked = st.button(
                "📦 Trigger B2B Liquidation Listing",
                use_container_width=True,
                help="List on B-Stock or similar marketplace",
            )
        with col3:
            reject_clicked = st.button(
                "⏸️ Reject / Keep Holding",
                use_container_width=True,
                help="Defer action; keep in warehouse",
            )

        action_taken = reroute_clicked or liquidate_clicked or reject_clicked
        if action_taken:
            if reroute_clicked and rec is not None:
                action_label = f"Approve Reroute to {rec['recommended_customer']}"
            elif liquidate_clicked:
                action_label = "Trigger B2B Liquidation Listing"
            else:
                action_label = "Reject / Keep Holding"
            append_audit_log(selected_lot, action_label)
            st.session_state.acted_lots.add(selected_lot)
            st.success(f"Action logged and audit trail updated for Lot {selected_lot}.")
            st.balloons()
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: ROI VALIDATION (MONTE CARLO)
# ---------------------------------------------------------------------------
def render_roi_validation_tab():
    """
    Render ROI Validation tab: histogram, percentiles, scatter, mean recovery rate.
    """
    sim_df = load_simulation_results()
    if sim_df is None or sim_df.empty:
        st.info(
            "Simulation results not found. Run `python scripts/run_simulation.py` to generate "
            "the Monte Carlo backtest data."
        )
        return

    avg_capital = sim_df["capital_recovered"].mean()
    max_capital = sim_df["capital_recovered"].max()
    avg_pct_reduction = sim_df["pct_reduction_writeoffs"].mean()
    p5 = sim_df["capital_recovered"].quantile(0.05)
    p50 = sim_df["capital_recovered"].quantile(0.50)
    p95 = sim_df["capital_recovered"].quantile(0.95)
    avg_recovery_rate = (
        sim_df["recovery_rate_mean"].mean() * 100
        if "recovery_rate_mean" in sim_df.columns
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Capital Recovered", f"${avg_capital:,.2f}")
    with col2:
        st.metric("P5 / P50 / P95 Capital Recovered", f"${p5:,.0f} / ${p50:,.0f} / ${p95:,.0f}")
    with col3:
        st.metric("Average % Reduction in Write-offs", f"{avg_pct_reduction:.1f}%")
    with col4:
        if avg_recovery_rate is not None:
            st.metric("Mean Recovery Rate (stochastic)", f"{avg_recovery_rate:.1f}%")

    st.divider()
    st.subheader("Distribution of Capital Recovered")
    st.bar_chart(sim_df[["iteration_id", "capital_recovered"]].set_index("iteration_id"))

    if "n_at_risk_lots" in sim_df.columns:
        st.subheader("At-Risk Lots vs Capital Recovered")
        st.scatter_chart(sim_df, x="n_at_risk_lots", y="capital_recovered")

    st.divider()
    st.markdown(
        "**About this simulation**  \n\n"
        "This simulation uses a **stochastic recovery model** (Beta distributions by channel and "
        "days-to-expiry), not a hardcoded 60%. Recovery rate varies by iteration. "
        "85% of reroute attempts succeed; 15% fail (truck unavailable, etc.)."
    )


# ---------------------------------------------------------------------------
# COLUMN MAPPING (flexible alias support for real Inventory Aging Reports)
# ---------------------------------------------------------------------------
SKU_ALIASES = {
    "sku_id": ["sku_id", "sku", "item", "item id", "product code"],
    "category": ["category", "product category"],
    "unit_cost": ["unit_cost", "unit cost", "cost"],
    "standard_shelf_life_days": ["standard_shelf_life_days", "shelf life", "shelf life days"],
}
LOT_ALIASES = {
    "lot_id": ["lot_id", "lot", "batch", "lot id"],
    "sku_id": ["sku_id", "sku", "item", "item id"],
    "location_id": ["location_id", "location", "warehouse", "dc"],
    "qty_on_hand": ["qty_on_hand", "qty", "quantity", "qty on hand"],
    "production_date": ["production_date", "production date", "prod date", "made date"],
    "expiry_date": ["expiry_date", "expiry date", "expiration", "best by"],
}
POLICY_ALIASES = {
    "customer_id": ["customer_id", "customer id"],
    "customer_name": ["customer_name", "customer name"],
    "required_rsl_pct": ["required_rsl_pct", "required rsl", "rsl minimum"],
    "transit_lead_time_days": ["transit_lead_time_days", "transit days", "lead time"],
}


def _normalize_columns(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    """Map columns to canonical names using case-insensitive alias matching."""
    df = df.copy()
    new_cols = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            match = next((c for c in df.columns if c.strip().lower() == alias.lower()), None)
            if match:
                new_cols[match] = canonical
                break
    df = df.rename(columns=new_cols)
    return df


DEFAULT_CUSTOMER_POLICIES = pd.DataFrame([
    {"customer_id": "CUST-001", "customer_name": "UNFI", "required_rsl_pct": 0.75, "transit_lead_time_days": 5},
    {"customer_id": "CUST-002", "customer_name": "Walmart", "required_rsl_pct": 0.60, "transit_lead_time_days": 7},
    {"customer_id": "CUST-003", "customer_name": "Discount_Partner", "required_rsl_pct": 0.10, "transit_lead_time_days": 3},
])


# ---------------------------------------------------------------------------
# TAB 3: RUN ON YOUR DATA
# ---------------------------------------------------------------------------
EDI_ALIASES = {
    "date": ["date", "report_date", "transaction_date"],
    "location_id": ["location_id", "location", "warehouse", "dc"],
    "sku_id": ["sku_id", "sku", "item", "item id"],
    "reported_qty_sold": ["reported_qty_sold", "qty_sold", "quantity", "units_sold"],
}


def render_upload_tab():
    """
    Upload CSV exports and run the risk engine. Validation errors/warnings displayed.
    Optional EDI 852 for signal integrity scoring.
    """
    st.subheader("Upload Your Data")
    st.markdown(
        "Upload your **Inventory Aging Report** and related exports. ReFlow will validate "
        "your data, run the Shelf-Life Gatekeeper, and show at-risk lots. "
        "Download templates below if your column names differ."
    )

    col1, col2 = st.columns(2)
    with col1:
        sku_file = st.file_uploader("SKU Master (required)", type=["csv"], key="upload_sku")
    with col2:
        lot_file = st.file_uploader("Lot Ledger / Inventory Aging (required)", type=["csv"], key="upload_lot")
    policy_file = st.file_uploader(
        "Customer Policies (optional)",
        type=["csv"],
        key="upload_policy",
        help="If omitted, we use defaults: UNFI 75%, Walmart 60%, Discount Partner 10%.",
    )
    edi_file = st.file_uploader(
        "EDI 852 Feed (optional)",
        type=["csv"],
        key="upload_edi",
        help="For signal integrity scoring. Columns: date, location_id, sku_id, reported_qty_sold.",
    )

    st.divider()
    st.subheader("Download Templates")
    with st.expander("Get CSV templates with required columns"):
        sku_template = pd.DataFrame(columns=["sku_id", "category", "unit_cost", "standard_shelf_life_days"])
        lot_template = pd.DataFrame(columns=["lot_id", "sku_id", "location_id", "qty_on_hand", "production_date", "expiry_date"])
        policy_template = pd.DataFrame(columns=["customer_id", "customer_name", "required_rsl_pct", "transit_lead_time_days"])
        st.download_button(
            "Download SKU template",
            sku_template.to_csv(index=False),
            file_name="sku_master_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Lot Ledger template",
            lot_template.to_csv(index=False),
            file_name="lot_ledger_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Customer Policies template",
            policy_template.to_csv(index=False),
            file_name="customer_policies_template.csv",
            mime="text/csv",
        )

    if not sku_file or not lot_file:
        st.info("Upload SKU Master and Lot Ledger to run the risk engine.")
        return

    try:
        sku = pd.read_csv(sku_file)
        lot = pd.read_csv(lot_file)
        sku = _normalize_columns(sku, SKU_ALIASES)
        lot = _normalize_columns(lot, LOT_ALIASES)

        required_sku = ["sku_id", "category", "unit_cost", "standard_shelf_life_days"]
        required_lot = ["lot_id", "sku_id", "location_id", "qty_on_hand", "production_date", "expiry_date"]
        missing_sku = [c for c in required_sku if c not in sku.columns]
        missing_lot = [c for c in required_lot if c not in lot.columns]

        if missing_sku or missing_lot:
            st.error(
                f"Missing columns. SKU: {missing_sku or 'OK'}. Lot: {missing_lot or 'OK'}. "
                "Rename columns to match our schema or use the templates."
            )
            return

        if policy_file:
            policy = pd.read_csv(policy_file)
            policy = _normalize_columns(policy, POLICY_ALIASES)
            required_policy = ["customer_id", "customer_name", "required_rsl_pct", "transit_lead_time_days"]
            missing = [c for c in required_policy if c not in policy.columns]
            if missing:
                st.error(f"Customer Policies missing columns: {missing}. Using defaults.")
                policy = DEFAULT_CUSTOMER_POLICIES
        else:
            policy = DEFAULT_CUSTOMER_POLICIES

        edi_852 = None
        if edi_file:
            edi_852 = pd.read_csv(edi_file)
            edi_852 = _normalize_columns(edi_852, EDI_ALIASES)
            if not all(c in edi_852.columns for c in ["date", "location_id", "sku_id", "reported_qty_sold"]):
                st.warning("EDI 852 missing required columns. Skipping signal integrity.")
                edi_852 = None

        df, recommendations, val_result = build_comprehensive_ledger(
            sku, policy, lot, edi_852=edi_852,
        )

        # Display validation errors (red) and warnings (yellow)
        if val_result.errors:
            st.error("**Validation Errors (processing halted):**")
            for e in val_result.errors:
                st.error(f"• {e.message}")
            return
        if val_result.warnings:
            st.warning("**Validation Warnings (proceeding with cleaned data):**")
            for w in val_result.warnings:
                st.warning(f"• {w.message}")
        if val_result.stats:
            st.caption(f"Data quality: {val_result.stats}")

        df_at_risk = df[df["is_at_risk"] == True]

        st.success(f"Processed {len(lot)} lots.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Lots at Risk", f"{len(df_at_risk):,}")
        with col2:
            st.metric("Financial Risk Exposure", f"${df_at_risk['financial_risk_exposure'].sum():,.2f}")
        with col3:
            st.metric("Total Inventory Value", f"${df['total_lot_value'].sum():,.2f}")
        with col4:
            total_drain = df_at_risk["drain_rate_daily"].sum() if "drain_rate_daily" in df_at_risk.columns else 0
            st.metric("Total Daily Drain", f"${total_drain:,.2f}")

        st.divider()
        st.subheader("At-Risk Lot Ledger")
        display_cols = ["lot_id", "sku_id", "location_id", "total_lot_value", "actual_days_remaining"]
        if "drain_rate_daily" in df_at_risk.columns:
            display_cols.extend(["drain_rate_daily", "projected_total_drain"])
        if "confidence_tier" in df_at_risk.columns:
            display_cols.append("confidence_tier")
        display_cols = [c for c in display_cols if c in df_at_risk.columns]
        if df_at_risk.empty:
            st.success("No lots at risk. All inventory meets customer shelf-life policies.")
        else:
            df_display = df_at_risk[display_cols].copy()
            if "confidence_tier" in df_display.columns:
                df_display["Signal"] = df_display["confidence_tier"].apply(_signal_indicator)
            st.dataframe(
                df_display.rename(columns={
                    "lot_id": "Lot ID",
                    "sku_id": "SKU ID",
                    "location_id": "Location ID",
                    "total_lot_value": "Total Lot Value",
                    "actual_days_remaining": "Days Left",
                    "drain_rate_daily": "Daily Drain ($)",
                    "projected_total_drain": "Projected Drain ($)",
                }),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download full At-Risk Ledger (CSV)",
            df.to_csv(index=False),
            file_name="at_risk_ledger_uploaded.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing files: {e}")
        st.exception(e)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    st.title("ReFlow AI | Action Inbox")
    st.caption("Governed liquidity management for mid-market FMCG supply chains")

    tab1, tab2, tab3 = st.tabs(["Live Action Inbox", "ROI Validation (Monte Carlo)", "Run on Your Data"])
    with tab1:
        render_action_inbox_tab()
    with tab2:
        render_roi_validation_tab()
    with tab3:
        render_upload_tab()


if __name__ == "__main__":
    main()
