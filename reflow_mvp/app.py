"""
ReFlow AI | Action Inbox (Phase 3)

PURPOSE:
    Governed Action Inbox UI for supply chain planners to review at-risk inventory
    and approve/deny agent-recommended actions. Human-in-the-loop for NIST AI RMF
    compliance—no autonomous execution without approval.

FEATURES:
    - Tab 1 (Live Action Inbox): At-risk lot grid, KPI metrics, action simulator
      (Reroute / Liquidate / Reject). Smart recommendations: Reroute only shown
      when lot is eligible for Discount Partner (eligible_discount_partner=True).
    - Tab 2 (ROI Validation): Monte Carlo backtest results—Average/Max Capital
      Recovered, % Reduction in Write-offs, distribution chart.
    - Audit trail: All actions logged to data/processed/audit_log.csv.
"""
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

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
    import pandas as pd
    df = pd.read_csv(LEDGER_PATH)
    # Filter to at-risk lots only
    df = df[df["is_at_risk"] == True].copy()
    return df


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
def render_action_inbox_tab():
    """
    Render Live Action Inbox: KPIs, at-risk lot grid, action simulator.
    Reroute button only shown when lot is eligible for Discount Partner (RSL >= 10%).
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

    if "acted_lots" not in st.session_state:
        st.session_state.acted_lots = set()

    df_remaining = df[~df["lot_id"].isin(st.session_state.acted_lots)]
    total_lots = len(df_remaining)
    total_exposure = df_remaining["financial_risk_exposure"].sum()
    strictest_policy = "UNFI - 75% RSL"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Lots at Risk", f"{total_lots:,}")
    with col2:
        st.metric("Total Financial Risk Exposure", f"${total_exposure:,.2f}")
    with col3:
        st.metric("Strictest Customer Policy", strictest_policy)

    st.divider()
    st.subheader("At-Risk Lot Ledger")
    df_display = df_remaining
    display_cols = ["lot_id", "sku_id", "location_id", "total_lot_value", "actual_days_remaining"]
    display_cols = [c for c in display_cols if c in df_display.columns]
    st.dataframe(
        df_display[display_cols].rename(columns={
            "lot_id": "Lot ID",
            "sku_id": "SKU ID",
            "location_id": "Location ID",
            "total_lot_value": "Total Lot Value",
            "actual_days_remaining": "Actual Days Remaining",
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
        eligible_discount = lot_row.get("eligible_discount_partner", False)
        if isinstance(eligible_discount, str):
            eligible_discount = eligible_discount.lower() in ("true", "1")

        col1, col2, col3 = st.columns(3)
        with col1:
            if eligible_discount:
                reroute_clicked = st.button(
                    "🔄 Reroute to Discount_Partner",
                    use_container_width=True,
                    help="Lot meets Discount Partner's 10% RSL minimum",
                )
            else:
                st.caption("Reroute: Not eligible (RSL below Discount Partner minimum)")
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
            if reroute_clicked:
                action_label = "Reroute to Discount_Partner"
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
    Render ROI Validation tab: metrics (avg/max capital recovered, % reduction),
    bar chart of capital recovered by iteration, explanatory markdown.
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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Capital Recovered", f"${avg_capital:,.2f}")
    with col2:
        st.metric("Maximum Capital Recovered", f"${max_capital:,.2f}")
    with col3:
        st.metric("Average % Reduction in Write-offs", f"{avg_pct_reduction:.1f}%")

    st.divider()
    st.subheader("Capital Recovered by Iteration")
    chart_data = sim_df[["iteration_id", "capital_recovered"]].set_index("iteration_id")
    st.bar_chart(chart_data)

    st.divider()
    st.markdown(
        "**About this simulation**  \n\n"
        "This simulation represents 100 stochastic supply chain environments, injecting random "
        "transit delays and demand shocks. ReFlow AI consistently recovers capital by preemptively "
        "routing inventory before it breaches strict retailer shelf-life thresholds."
    )


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    st.title("ReFlow AI | Action Inbox")
    st.caption("Governed liquidity management for mid-market FMCG supply chains")

    tab1, tab2 = st.tabs(["Live Action Inbox", "ROI Validation (Monte Carlo)"])
    with tab1:
        render_action_inbox_tab()
    with tab2:
        render_roi_validation_tab()


if __name__ == "__main__":
    main()
