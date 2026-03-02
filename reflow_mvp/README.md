# ReFlow AI | Governed Liquidity Engine for FMCG

**Treating inventory as a decaying financial asset—not a static number.**

ReFlow AI is a governed, closed-loop control system that stops the physical shipment of commercially ineligible goods and autonomously reroutes them to preserve capital. Instead of trying to build a better crystal ball for consumer demand, we act as a **Shelf-Life Gatekeeper**: we manage the drain rate of every pallet and intercept doomed shipments before they hit the dock.

---

## The Problem: "The Double Loss"

Mid-market FMCG and CPG brands ($50M–$200M) bleed millions annually to **Unproductive Inventory (UPI)**—not because they lack forecasting tools, but because their systems optimize against distorted signals and ignore the **"hard clock"** of wholesale supply chains.

### Retailer Shelf-Life Minimums

Distributors and retailers enforce strict receiving policies. If a shipment arrives with less than the required **Remaining Shelf Life (RSL)** at receipt, it is **rejected at the dock**:

- **UNFI**: 75% RSL at receipt (product must have 75% of its shelf life remaining)
- **Walmart**: ~60% RSL
- **Discount Partner / Liquidators**: ~10% RSL (lenient)

### Why Standard Forecasting Fails

1. **Unit of planning mismatch**: Legacy systems plan at SKU/week aggregates. Shelf-life risk is **lot-specific** and **customer-policy-specific**. A pallet can be operationally feasible yet commercially ineligible on arrival.

2. **The Double Loss**: When a lot is rejected, the brand suffers (a) total write-off of the inventory, plus (b) return freight logistics. No forecast accuracy can undo that.

3. **Bullwhip distortion**: EDI 852 and order data are poisoned by demand signal processing, rationing games, and promotion spikes. Planning off distorted signals predicts the distortion, not the demand.

---

## The Solution: Agentic Control

We shift from **"predicting demand"** to **"managing risk"**:

- **FEFO Router**: Replaces static FIFO. Tracks inventory at lot/batch level. Calculates transit lead time against the specific shelf-life policy of the destination retailer. If a lot is mathematically guaranteed to be rejected, we flag it and prepare an alternative routing plan.

- **Governed Action Inbox**: Every recommendation requires human approval. NIST AI RMF–aligned: auditable, traceable, no autonomous execution without explicit sign-off.

- **Signal Integrity Scorer**: Z-score anomaly detection on EDI 852 feeds; confidence tiers (HIGH/MEDIUM/LOW) flag phantom demand spikes.

---

## The Architecture: 5 Phases

| Phase | Deliverable | Description |
|-------|-------------|-------------|
| **Phase 1** | Stochastic Data Generation | Bias-free synthetic data using Gaussian (production dates) and Poisson (demand) distributions. No hardcoded at-risk lots—failures emerge from statistical tails. Target: 1.5–3% organic failure rate (FMCG benchmark). Schema validation assertion at end of `generate_all()`. |
| **Phase 2** | At-Risk Engine | Shelf-Life Gatekeeper + FEFO Router + Drain Rate + Signal Integrity. Validates inputs (schema.py), builds eligibility matrix, recommends reroutes, computes daily economic drain. Outputs comprehensive ledger with drain_rate_daily, projected_total_drain, signal_confidence. |
| **Phase 3** | Action Inbox | Streamlit UI: at-risk lot grid sorted by drain rate, FEFO reroute recommendations, "Approve Recommended Action" button, Total Daily Capital Drain KPI. **Run on Your Data** tab: validation errors/warnings, optional EDI 852 for signal scoring. |
| **Phase 4** | Monte Carlo Validation | 100 stochastic warehouse scenarios. **Stochastic recovery model** (Beta by channel + days-to-expiry), 85% intervention success rate. P5/P50/P95 percentiles, scatter plot n_at_risk vs capital_recovered. Mean recovery rate varies by iteration. |

---

## Mathematical Validation

The Monte Carlo backtest runs **100 distinct warehouse states** (different seeds → different transit delays, demand shocks, lot ages). Uses a **stochastic recovery model** (Beta distributions by channel and days-to-expiry; 85% intervention success rate)—not a hardcoded 60%. Results:

- **~50–60% mean recovery rate** (varies by iteration; fire-sale lots get lower recovery)
- **P5 / P50 / P95** percentiles of capital recovered
- **Scatter plot**: n_at_risk_lots vs capital_recovered (relationship is not linear)

This is **not a hardcoded demo**. At-risk lots emerge organically; recovery rates are sampled from industry-calibrated distributions.

---

## Quickstart

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository (or navigate to project root)
cd reflow_mvp

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies**: `pandas`, `numpy`, `streamlit`, `faker` (pure Python validation; no pandera/pydantic)

### Run the Pipeline

```bash
# Phase 1: Generate synthetic data
python scripts/generate_synthetic_data.py

# Phase 2: Build at-risk ledger
python scripts/build_risk_engine.py

# Phase 4: Run Monte Carlo backtest (proves ROI)
python scripts/run_simulation.py

# Phase 3: Launch Action Inbox UI
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Run on Your Data (CSV Upload)

You can run the risk engine on **your own data** without generating synthetic data:

1. Launch the app: `streamlit run app.py`
2. Open the **"Run on Your Data"** tab
3. Upload two required CSVs:
   - **SKU Master**: `sku_id`, `category`, `unit_cost`, `standard_shelf_life_days`
   - **Lot Ledger** (Inventory Aging Report): `lot_id`, `sku_id`, `location_id`, `qty_on_hand`, `production_date`, `expiry_date`
4. Optionally upload **Customer Policies**; if omitted, defaults (UNFI 75%, Walmart 60%, Discount Partner 10%) are used
5. Optionally upload **EDI 852** for signal integrity scoring (confidence tiers per location-SKU)
6. Download CSV templates from the expander if your column names differ

**Validation**: Red errors halt processing (missing columns, zero valid rows). Yellow warnings (nulls dropped, duplicates deduped, orphan SKUs) are logged; processing continues with cleaned data.

**Use case**: During pilot discovery, ask prospects to export their Inventory Aging Report. Upload it to show them at-risk lots, drain rate, and FEFO reroute recommendations—no ERP integration required.

---

## Directory Structure

```
reflow_mvp/
├── data/
│   ├── raw_synthetic/       # Phase 1 outputs (sku_master, customer_policies, lot_ledger, edi_852_feed)
│   └── processed/           # Phase 2+ outputs (at_risk_ledger, reroute_recommendations, audit_log, simulation_results)
├── scripts/
│   ├── schema.py                    # Schema validation (Task 1)
│   ├── fefo_engine.py               # FEFO eligibility matrix + reroute recommender (Task 2)
│   ├── drain_rate.py                # Daily economic drain calculator (Task 4)
│   ├── signal_integrity.py          # EDI 852 z-score anomaly detection (Task 5)
│   ├── generate_synthetic_data.py   # Phase 1: Stochastic data generation
│   ├── build_risk_engine.py         # Phase 2: Orchestrates validation → FEFO → drain → signal
│   └── run_simulation.py            # Phase 4: Monte Carlo with stochastic recovery
├── app.py                   # Phase 3: Streamlit Action Inbox
├── requirements.txt
└── README.md
```

---

## Key Concepts

| Term | Definition |
|------|------------|
| **RSL** | Remaining Shelf Life = (expiry_date - today) / (expiry_date - production_date) |
| **At-Risk Lot** | Ineligible for strictest customer (e.g., UNFI 75%) but not expired; can be rerouted |
| **FEFO** | First Expiry, First Out—route earliest-expiring lots to highest-velocity nodes that can accept them |
| **Bullwhip Effect** | Order variance exceeds sales variance; upstream signals become distorted |

---

## License

Proprietary. All rights reserved.
