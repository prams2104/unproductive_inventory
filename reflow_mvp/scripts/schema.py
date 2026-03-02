#!/usr/bin/env python3
"""
ReFlow AI — Schema Validation Layer

PURPOSE:
    Validates input DataFrames (SKU Master, Lot Ledger, Customer Policies) before
    the risk engine processes them. Prevents garbage-in-garbage-out: blank expiry
    dates, duplicate lot IDs, negative quantities, or non-numeric costs would
    otherwise produce wrong results or crashes. Critical for pilot readiness.

FMCG CONTEXT:
    Real prospect data from ERP exports often has nulls, duplicates, and format
    inconsistencies. This layer classifies issues as ERRORS (halt) vs WARNINGS
    (log and continue with cleaned data). Referential integrity (lot_ledger.sku_id
    must exist in sku_master) prevents orphan lots from skewing exposure metrics.

NO EXTERNAL DEPENDENCIES: Pure Python + pandas. Keeps requirements.txt lean.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# CANONICAL SCHEMAS (column definitions for validation)
# ---------------------------------------------------------------------------
SKU_MASTER_COLUMNS = ["sku_id", "category", "unit_cost", "standard_shelf_life_days"]
LOT_LEDGER_COLUMNS = [
    "lot_id", "sku_id", "location_id", "qty_on_hand",
    "production_date", "expiry_date",
]
CUSTOMER_POLICIES_COLUMNS = [
    "customer_id", "customer_name", "required_rsl_pct", "transit_lead_time_days",
]


@dataclass
class ValidationError:
    """Critical issue that halts processing. FMCG: missing data = cannot trust output."""
    message: str
    column: str | None = None
    row_index: int | None = None


@dataclass
class ValidationWarning:
    """Non-fatal issue: we can proceed with cleaned data. FMCG: log for audit."""
    message: str
    column: str | None = None
    row_index: int | None = None
    rows_affected: int | None = None


@dataclass
class ValidationResult:
    """
    Result of validate_dataframe() or validate_referential_integrity().

    is_valid: False if errors exist (halt processing).
    errors: Critical issues (missing columns, all null, zero valid rows).
    warnings: Non-fatal (nulls dropped, negatives corrected, duplicates deduped).
    cleaned_df: Recoverable DataFrame after applying corrections; None if unrecoverable.
    stats: Summary counts for audit (rows_input, rows_valid, rows_dropped, etc.).
    """
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)
    cleaned_df: pd.DataFrame | None = None
    stats: dict = field(default_factory=dict)


def _to_datetime_safe(series: pd.Series) -> pd.Series:
    """Parse datetime column; invalid values become NaT.

    Uses format='mixed' so a column containing multiple date formats (e.g.
    '2025-01-15', '01/15/2025', '15-Jan-2025') does not coerce all rows to NaT
    after latching onto the first row's format — a real ERP export failure mode.
    """
    return pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")


def validate_dataframe(
    df: pd.DataFrame,
    schema_name: Literal["sku_master", "lot_ledger", "customer_policies"],
) -> ValidationResult:
    """
    Validate a DataFrame against its canonical schema.

    ERRORS (halt processing):
        - Missing required columns
        - All values null in a required column
        - Zero valid rows after cleaning

    WARNINGS (log and continue):
        - Individual null rows dropped
        - Negative quantities set to 0
        - Duplicate lot_ids (keep first)
        - expiry_date <= production_date (drop row)
        - sku_id in lot_ledger not in sku_master (handled by validate_referential_integrity)

    Args:
        df: Input DataFrame (may have alias column names—caller must normalize first).
        schema_name: Which schema to validate against.

    Returns:
        ValidationResult with is_valid, errors, warnings, cleaned_df, stats.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []
    stats: dict = {"rows_input": len(df)}

    if schema_name == "sku_master":
        required = SKU_MASTER_COLUMNS
    elif schema_name == "lot_ledger":
        required = LOT_LEDGER_COLUMNS
    elif schema_name == "customer_policies":
        required = CUSTOMER_POLICIES_COLUMNS
    else:
        errors.append(ValidationError(message=f"Unknown schema: {schema_name}"))
        return ValidationResult(is_valid=False, errors=errors, stats=stats)

    # ERROR: Missing required columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(
            ValidationError(
                message=f"Missing required columns: {missing}. FMCG: cannot compute exposure without these.",
                column=",".join(missing),
            )
        )
        return ValidationResult(is_valid=False, errors=errors, stats=stats)

    cleaned = df[required].copy()

    # ERROR: All values null in required column
    for col in required:
        if cleaned[col].isna().all():
            errors.append(
                ValidationError(
                    message=f"Column '{col}' has all null values. FMCG: no valid data to process.",
                    column=col,
                )
            )
            return ValidationResult(is_valid=False, errors=errors, stats=stats)

    # Schema-specific validation and cleaning
    if schema_name == "sku_master":
        cleaned, w = _validate_sku_master(cleaned)
        warnings.extend(w)
    elif schema_name == "lot_ledger":
        cleaned, w = _validate_lot_ledger(cleaned)
        warnings.extend(w)
    elif schema_name == "customer_policies":
        cleaned, w = _validate_customer_policies(cleaned)
        warnings.extend(w)

    stats["rows_valid"] = len(cleaned)
    stats["rows_dropped"] = stats["rows_input"] - stats["rows_valid"]

    # ERROR: Zero valid rows after cleaning
    if len(cleaned) == 0:
        errors.append(
            ValidationError(
                message="Zero valid rows after cleaning. FMCG: cannot produce meaningful output.",
            )
        )
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            cleaned_df=None,
            stats=stats,
        )

    return ValidationResult(
        is_valid=True,
        errors=errors,
        warnings=warnings,
        cleaned_df=cleaned,
        stats=stats,
    )


def _validate_sku_master(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ValidationWarning]]:
    """
    SKU Master rules:
    - sku_id: str, not null, unique
    - category: str, not null
    - unit_cost: float, > 0
    - standard_shelf_life_days: int, > 0

    FMCG: SKU master drives value and shelf-life calculations. Invalid costs or
    shelf lives would corrupt financial exposure.
    """
    warnings: list[ValidationWarning] = []
    df = df.copy()

    # Drop null sku_id or category
    before = len(df)
    df = df.dropna(subset=["sku_id", "category"])
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped rows with null sku_id or category",
                rows_affected=before - len(df),
            )
        )

    # Deduplicate sku_id (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["sku_id"], keep="first")
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped duplicate sku_id rows (kept first)",
                rows_affected=before - len(df),
            )
        )

    # unit_cost: coerce to float, set invalid/negative to NaN then drop
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
    invalid_cost = (df["unit_cost"].isna()) | (df["unit_cost"] <= 0)
    if invalid_cost.any():
        n = invalid_cost.sum()
        warnings.append(
            ValidationWarning(
                message=f"Dropped {n} rows with invalid unit_cost (must be > 0)",
                column="unit_cost",
                rows_affected=int(n),
            )
        )
        df = df[~invalid_cost]

    # standard_shelf_life_days: coerce to int, must be > 0
    df["standard_shelf_life_days"] = pd.to_numeric(df["standard_shelf_life_days"], errors="coerce")
    invalid_shelf = (df["standard_shelf_life_days"].isna()) | (df["standard_shelf_life_days"] <= 0)
    if invalid_shelf.any():
        n = invalid_shelf.sum()
        warnings.append(
            ValidationWarning(
                message=f"Dropped {n} rows with invalid standard_shelf_life_days (must be > 0)",
                column="standard_shelf_life_days",
                rows_affected=int(n),
            )
        )
        df = df[~invalid_shelf]

    df["standard_shelf_life_days"] = df["standard_shelf_life_days"].astype(int)
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    return df, warnings


def _validate_lot_ledger(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ValidationWarning]]:
    """
    Lot Ledger rules:
    - lot_id: str, not null, unique
    - sku_id: str, not null (referential integrity checked separately)
    - location_id: str, not null
    - qty_on_hand: int, >= 0 (negative → set to 0)
    - production_date: datetime, not null, <= today
    - expiry_date: datetime, not null, > production_date

    FMCG: Negative qty is a data error; we correct to 0. Expiry <= production
    indicates bad data; drop row.
    """
    warnings: list[ValidationWarning] = []
    df = df.copy()

    # Parse dates
    df["production_date"] = _to_datetime_safe(df["production_date"])
    df["expiry_date"] = _to_datetime_safe(df["expiry_date"])

    # Drop null lot_id, sku_id, location_id
    before = len(df)
    df = df.dropna(subset=["lot_id", "sku_id", "location_id"])
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped rows with null lot_id, sku_id, or location_id",
                rows_affected=before - len(df),
            )
        )

    # Deduplicate lot_id (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["lot_id"], keep="first")
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped duplicate lot_id rows (kept first)",
                rows_affected=before - len(df),
            )
        )

    # qty_on_hand: coerce to int, negative → 0
    df["qty_on_hand"] = pd.to_numeric(df["qty_on_hand"], errors="coerce").fillna(0)
    neg_qty = df["qty_on_hand"] < 0
    if neg_qty.any():
        n = neg_qty.sum()
        warnings.append(
            ValidationWarning(
                message=f"Set {int(n)} negative qty_on_hand values to 0",
                column="qty_on_hand",
                rows_affected=int(n),
            )
        )
        df.loc[neg_qty, "qty_on_hand"] = 0
    df["qty_on_hand"] = df["qty_on_hand"].astype(int)

    # Drop null dates
    before = len(df)
    df = df.dropna(subset=["production_date", "expiry_date"])
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped rows with null production_date or expiry_date",
                rows_affected=before - len(df),
            )
        )

    # production_date <= today (warning only; we allow future for synthetic data)
    today = pd.Timestamp(datetime.now().date())
    future_prod = df["production_date"] > today
    if future_prod.any():
        n = future_prod.sum()
        warnings.append(
            ValidationWarning(
                message=f"{int(n)} rows have production_date > today (unusual; kept for processing)",
                column="production_date",
                rows_affected=int(n),
            )
        )

    # expiry_date > production_date; drop invalid
    before = len(df)
    invalid_expiry = df["expiry_date"] <= df["production_date"]
    df = df[~invalid_expiry]
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped rows where expiry_date <= production_date",
                rows_affected=before - len(df),
            )
        )

    df["lot_id"] = df["lot_id"].astype(str).str.strip()
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["location_id"] = df["location_id"].astype(str).str.strip()

    return df, warnings


def _validate_customer_policies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ValidationWarning]]:
    """
    Customer Policies rules:
    - customer_id: str, not null, unique
    - customer_name: str, not null
    - required_rsl_pct: float, 0.0 <= x <= 1.0
    - transit_lead_time_days: int, > 0

    FMCG: RSL outside [0,1] or transit <= 0 would break shelf-life gatekeeper.
    """
    warnings: list[ValidationWarning] = []
    df = df.copy()

    # Drop null
    before = len(df)
    df = df.dropna(subset=["customer_id", "customer_name"])
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped rows with null customer_id or customer_name",
                rows_affected=before - len(df),
            )
        )

    # Deduplicate customer_id
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    if len(df) < before:
        warnings.append(
            ValidationWarning(
                message="Dropped duplicate customer_id rows (kept first)",
                rows_affected=before - len(df),
            )
        )

    # required_rsl_pct: 0.0 to 1.0
    df["required_rsl_pct"] = pd.to_numeric(df["required_rsl_pct"], errors="coerce")
    invalid_rsl = (df["required_rsl_pct"].isna()) | (df["required_rsl_pct"] < 0) | (df["required_rsl_pct"] > 1)
    if invalid_rsl.any():
        n = invalid_rsl.sum()
        warnings.append(
            ValidationWarning(
                message=f"Dropped {int(n)} rows with required_rsl_pct outside [0, 1]",
                column="required_rsl_pct",
                rows_affected=int(n),
            )
        )
        df = df[~invalid_rsl]

    # transit_lead_time_days: > 0
    df["transit_lead_time_days"] = pd.to_numeric(df["transit_lead_time_days"], errors="coerce")
    invalid_transit = (df["transit_lead_time_days"].isna()) | (df["transit_lead_time_days"] <= 0)
    if invalid_transit.any():
        n = invalid_transit.sum()
        warnings.append(
            ValidationWarning(
                message=f"Dropped {int(n)} rows with transit_lead_time_days <= 0",
                column="transit_lead_time_days",
                rows_affected=int(n),
            )
        )
        df = df[~invalid_transit]

    df["transit_lead_time_days"] = df["transit_lead_time_days"].astype(int)
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["customer_name"] = df["customer_name"].astype(str).str.strip()

    return df, warnings


def validate_referential_integrity(
    lot_ledger: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> ValidationResult:
    """
    Check that every sku_id in lot_ledger exists in sku_master.

    FMCG: Orphan lots (SKU not in master) cannot be valued or shelf-life checked.
    We drop them and log as warnings.

    Args:
        lot_ledger: Validated lot ledger (after validate_dataframe).
        sku_master: Validated SKU master.

    Returns:
        ValidationResult with cleaned lot_ledger (orphans removed), warnings for orphan SKUs.
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []
    valid_skus = set(sku_master["sku_id"].astype(str).str.strip())
    lot_ledger = lot_ledger.copy()
    lot_ledger["sku_id"] = lot_ledger["sku_id"].astype(str).str.strip()
    rows_input = len(lot_ledger)

    orphans = ~lot_ledger["sku_id"].isin(valid_skus)
    n_orphans = orphans.sum()
    if n_orphans > 0:
        orphan_skus = lot_ledger.loc[orphans, "sku_id"].unique().tolist()
        warnings.append(
            ValidationWarning(
                message=f"Dropped {int(n_orphans)} lot rows with sku_id not in SKU Master (orphans: {orphan_skus[:10]}{'...' if len(orphan_skus) > 10 else ''})",
                column="sku_id",
                rows_affected=int(n_orphans),
            )
        )
        lot_ledger = lot_ledger[~orphans]

    stats = {
        "rows_input": rows_input,
        "rows_valid": len(lot_ledger),
        "rows_dropped": int(n_orphans),
    }
    return ValidationResult(
        is_valid=True,
        errors=errors,
        warnings=warnings,
        cleaned_df=lot_ledger if len(lot_ledger) > 0 else None,
        stats=stats,
    )


if __name__ == "__main__":
    """Demonstrate schema validation on synthetic data."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw_synthetic"

    print("ReFlow AI — Schema Validation (standalone test)")
    print("=" * 55)

    sku = pd.read_csv(raw_dir / "sku_master.csv")
    lot = pd.read_csv(raw_dir / "lot_ledger.csv")
    policy = pd.read_csv(raw_dir / "customer_policies.csv")

    for name, df, schema in [
        ("SKU Master", sku, "sku_master"),
        ("Lot Ledger", lot, "lot_ledger"),
        ("Customer Policies", policy, "customer_policies"),
    ]:
        result = validate_dataframe(df, schema)
        print(f"\n{name}: is_valid={result.is_valid}, stats={result.stats}")
        for e in result.errors:
            print(f"  ERROR: {e.message}")
        for w in result.warnings:
            print(f"  WARN:  {w.message}")

    # Referential integrity
    sku_result = validate_dataframe(sku, "sku_master")
    lot_result = validate_dataframe(lot, "lot_ledger")
    if sku_result.cleaned_df is not None and lot_result.cleaned_df is not None:
        ref_result = validate_referential_integrity(lot_result.cleaned_df, sku_result.cleaned_df)
        print(f"\nReferential integrity: is_valid={ref_result.is_valid}, stats={ref_result.stats}")
        for w in ref_result.warnings:
            print(f"  WARN:  {w.message}")

    print("\nDone.")
