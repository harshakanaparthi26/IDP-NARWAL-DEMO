# Script 02 — `tsg_benchmark.py`

---

## Opening

Alright, now we're going into `tsg_benchmark.py`. This is the core utility layer of the entire benchmark pipeline. Everything else — the orchestrator, the Streamlit UI — ultimately calls into this file. The design principle here is simple: pure functions, no side effects, no writes. Every function takes inputs, returns outputs, and nothing else happens. That makes this file independently testable and completely safe to call from multiple places, which is exactly what we do — both `benchmark_main_pipeline.py` and the Streamlit UI call into this directly.

Let me walk through it section by section.

---

## Module Docstring

```python
"""
Autonomous benchmark utilities for the Benchmark Comparison service.

Provides:
- Safe numeric parsing
- S3/local CSV loaders
- Dropdown helpers for cascading selections
- Effective rate calculation (decimal)
- Benchmark comparison (decimal)
- Basis points conversion + ISV quote suggestions

No side-effects (no writes). Designed to be used by a service-level
`benchmark_main_pipeline.py` or higher-level orchestrators.
"""
```

The docstring is worth reading because it tells you exactly what this file provides and — importantly — what it doesn't do. No writes, no side effects. This is the contract. If you're a developer picking this up in the future, this tells you you can import anything from here without worrying about unexpected behavior.

---

## Imports

```python
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd

import settings
```

We're using `boto3` to talk to S3 for the benchmark CSV, `pandas` for all the DataFrame operations, and `settings` for the S3 bucket name and key. The `BytesIO` import is interesting — we'll see why we need that when we get to the loader. `re` is for the string sanitization in the numeric parser.

---

## `_to_float` — Safe Numeric Parser

```python
def _to_float(val) -> Optional[float]:
    """
    Convert possibly messy numeric input to float.

    Accepted:
    - int/float → float(val)
    - str like '$1,234.56' or ' -245 ' → 1234.56 / -245.0
    - otherwise → None
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = re.sub(r"[^0-9.\-]", "", val.strip())
        try:
            return float(s) if s not in ("", "-", ".", "-.", ".-") else None
        except Exception:
            return None
    return None
```

This little helper does a lot of work quietly throughout the file. The reason it exists is that we're dealing with data extracted from merchant statements by an LLM. That data is messy. Values come back as strings like `"$1,234.56"` or `"245 bps"` or just `None` if the extraction missed it. We need a parser that can handle all of that without crashing.

The regex `[^0-9.\-]` strips everything except digits, dots, and minus signs. Then we have a guard against degenerate strings — empty string, just a dash, just a dot — before attempting the float conversion. The whole thing is wrapped in a try-except so it always returns either a float or None, never raises. You'll see `_to_float` called throughout this file whenever we touch a value that came from entities or from the benchmark CSV.

---

## `load_isv_table` — Local CSV Loader

```python
def load_isv_table(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        csv_path = path or "/home/ec2-user/SageMaker/IDP_MAIN_APPLICATION/ISVdetails.csv"
        df = pd.read_csv(csv_path)

        required = ["ISV Name", "ISV Benchmark"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        df["ISV Name"] = df["ISV Name"].astype(str).str.strip()
        df["ISV Benchmark"] = pd.to_numeric(df["ISV Benchmark"], errors="coerce")

        logger.info("[Benchmark] Loaded %s ISV rows from %s", len(df), csv_path)
        return df

    except Exception as e:
        logger.warning("[Benchmark] Could not load ISV table: %s", e)
        return None
```

This loads the ISV lookup table from a local CSV on the SageMaker instance. As we saw in the overview, that CSV has two columns — `ISV Name` and `ISV Benchmark`. The function accepts an optional path override, which is important — the Streamlit UI passes an explicit path, and the pipeline orchestrator calls it with no argument to use the default.

We validate the columns before doing anything else. If the CSV is missing either expected column, we raise immediately with a clear message rather than letting a KeyError surface somewhere deeper. Then we normalize — ISV names get stripped of whitespace, and the benchmark values get coerced to numeric. The `errors="coerce"` on the numeric conversion means any value that can't be parsed becomes NaN rather than crashing, which is consistent with the defensive style throughout this file.

Everything is wrapped in a try-except that returns None on failure. That pattern repeats across all the loaders here — never raise, always return None and log a warning. The callers check for None before proceeding.

---

## `load_benchmarks` — S3 CSV Loader

```python
def load_benchmarks(
    *,
    s3_client: Optional[Any] = None,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    try:
        s3 = s3_client or boto3.client("s3")
        b = bucket or settings.S3_SNOWFLAKE_BUCKET
        k = key or settings.TSG_CSV_KEY

        obj = s3.get_object(Bucket=b, Key=k)
        df = pd.read_csv(BytesIO(obj["Body"].read()))

        required_cols = ["INDUSTRY", "BENCHMARK", "VOLUME_TIER", "PRICING_TYPE", "REGION"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        df["INDUSTRY"] = df["INDUSTRY"].astype(str).str.strip()
        df["BENCHMARK"] = pd.to_numeric(df["BENCHMARK"], errors="coerce")
        df["VOLUME_TIER"] = df["VOLUME_TIER"].astype(str).str.strip()
        df["PRICING_TYPE"] = df["PRICING_TYPE"].astype(str).str.strip()
        df["REGION"] = df["REGION"].astype(str).str.strip()

        logger.info("[Benchmark] Loaded %s benchmark rows from s3://%s/%s", len(df), b, k)
        return df

    except Exception as e:
        logger.warning("[Benchmark] Could not load benchmarks: %s", e)
        return None
```

This is the S3 loader for the main benchmark table. The structure is similar to `load_isv_table`, but there are a few things worth pointing out.

The function signature uses keyword-only arguments — that `*` at the start — with three optional overrides: `s3_client`, `bucket`, and `key`. This is a dependency injection pattern. It means in testing you can pass in a mock S3 client and a test bucket without touching the real infrastructure. In production, all three default to the real values from `settings`. That's a clean design for something that talks to AWS.

The `BytesIO` wrapper is what makes S3 streaming work with pandas. `s3.get_object` returns a streaming response body. pandas `read_csv` expects a file-like object. `BytesIO` bridges those two by reading the raw bytes into memory and wrapping them in a file-like interface. It's a small detail but it's the right way to do this — no temp files, no disk writes.

We validate and normalize the same way as the ISV loader. Five required columns, all strings get stripped, the benchmark decimal gets coerced to numeric. The expected format is decimal — so `0.0245` means 245 basis points. We'll see the conversion to BPS later.

---

## Dropdown Helpers — Cascading Filters

```python
def get_industries(bench_df: pd.DataFrame) -> List[str]:
    if bench_df is None or bench_df.empty or "INDUSTRY" not in bench_df.columns:
        return []
    return sorted(
        bench_df["INDUSTRY"].dropna().astype(str).str.strip().unique().tolist()
    )
```

```python
def get_volume_tiers(bench_df: pd.DataFrame, industry: str) -> List[str]:
    mg = (industry or "").strip()
    if not mg or "INDUSTRY" not in bench_df.columns or "VOLUME_TIER" not in bench_df.columns:
        return []
    df = bench_df[bench_df["INDUSTRY"].astype(str).str.strip() == mg].copy()
    if df.empty:
        return []
    return sorted(
        df["VOLUME_TIER"].dropna().astype(str).str.strip().unique().tolist()
    )
```

```python
def get_pricing_types(
    bench_df: pd.DataFrame,
    industry: Optional[str],
    volume_tier: Optional[str],
) -> List[str]:
    ...
```

```python
def get_regions(
    bench_df: pd.DataFrame,
    industry: Optional[str],
    volume_tier: Optional[str],
    pricing_type: Optional[str],
) -> List[str]:
    ...
```

These four functions power the cascading sidebar dropdowns in the Streamlit UI. The design is intentional — each function takes the filters selected so far and returns only the valid options for the next level down. So `get_volume_tiers` only returns tiers that exist for the selected industry. `get_pricing_types` only returns pricing types that exist for that industry and tier combination. And `get_regions` only returns regions that exist for all three of the above.

This prevents the user from selecting a combination that has no corresponding row in the benchmark table. It's a data-driven filter cascade — the options come directly from what's actually in the CSV, not from a hardcoded list somewhere. If you add a new industry to the benchmark table on S3, it just appears automatically in the dropdown the next time the page loads.

Every function guards against a None or empty DataFrame at the top and returns an empty list rather than crashing. The Streamlit UI handles an empty list gracefully by showing nothing.

---

## `compute_effective_rate`

```python
def compute_effective_rate(entities: Dict[str, Any]) -> Dict[str, Any]:
    if not entities:
        return {
            "effective_rate_raw": None,
            "notes": "No entities passed to compute_effective_rate",
        }

    ta = _to_float(entities.get("total_amount"))
    tf = _to_float(entities.get("total_fees"))
    notes: List[str] = []
    er: Optional[float] = None

    if ta is None:
        notes.append("total_amount missing")
    elif ta == 0:
        notes.append("total_amount is 0 - division prevented")

    if tf is None:
        notes.append("total_fees missing")

    if ta and ta != 0 and tf is not None:
        er = abs(tf) / abs(ta)

    return {"effective_rate_raw": er, "notes": " | ".join(notes)}
```

This is the core financial formula. Effective rate is simply total fees divided by total amount — what percentage of the merchant's total transaction volume went to fees. We use `abs()` on both values because depending on how the LLM extracted them, either value could come back as negative. The absolute value makes the math robust to that.

Two important guards here. First, we prevent division by zero explicitly — `ta == 0` is checked separately from `ta is None` so we can log a specific note about it rather than just a generic missing value message. Second, we use `_to_float` on both inputs, which handles the messy string formats we talked about earlier.

The function returns a dictionary with both the result and a `notes` string. That notes string accumulates any issues encountered — missing fields, zero amounts — and gets propagated all the way up through the pipeline into the final response. That's how we audit why a benchmark comparison came back empty without having to dig through logs.

---

## `to_bps` — Basis Points Conversion

```python
def to_bps(rate: Optional[float]) -> Optional[float]:
    """
    Convert a decimal rate to basis points (bps).
    Example: 0.0245 -> 245.0
    """
    if rate is None:
        return None
    try:
        return round(float(rate) * 10000, 2)
    except Exception:
        return None
```

Short and straightforward. One basis point is one hundredth of a percent, so one percent is 100 basis points, and a decimal like `0.0245` is `245` basis points. Multiply by ten thousand, round to two decimal places. We keep all internal calculations in decimal throughout the pipeline and only convert to BPS at display time. That's why this function appears mostly at the output stage, not inside the core logic.

---

## `make_isv_quotes` — Three-Tier Quote Generator

```python
def make_isv_quotes(eff_bps: float, isv_bps: float) -> Dict[str, float]:
    """
    Create 3 quote options labeled:
    - Low Profitability
    - Mid Profitability
    - High Profitability

    Uses GAP-based interpolation regardless of ER vs ISV:
    - Low  = 80% of gap toward ISV
    - Mid  = 50% of gap toward ISV
    - High = 20% of gap toward ISV
    """
    if eff_bps is None or isv_bps is None:
        return {}

    gap = eff_bps - isv_bps

    q_best = max(eff_bps - 0.80 * gap, 0)
    q_better = max(eff_bps - 0.50 * gap, 0)
    q_great = max(eff_bps - 0.20 * gap, 0)

    return {
        "Low Profitability": round(q_best, 2),
        "Mid Profitability": round(q_better, 2),
        "High Profitability": round(q_great, 2),
    }
```

This is the function that generates the actual quotes. The math is worth understanding clearly. We compute the gap — that's the merchant's effective rate in BPS minus the ISV's benchmark in BPS. Then we generate three quotes by moving different percentages of the way from the merchant's rate toward the ISV's rate.

Low Profitability moves 80% of the gap toward the ISV — so you're quoting close to the ISV's rate, which means less margin for us. High Profitability moves only 20% of the gap — you're staying closer to the merchant's current effective rate, which means more margin. Mid Profitability is the 50% midpoint between the two.

The `max(..., 0)` floor is important — it prevents us from ever quoting a negative rate, which would be nonsensical. And the key design decision here is that this works regardless of whether the merchant's effective rate is above or below the ISV benchmark. The gap can be positive or negative, and the interpolation handles both cases correctly. That's called out explicitly in the docstring — "regardless of ER vs ISV" — because it's a non-obvious property of this formula that a future developer might question.

---

## `compare_with_benchmark`

```python
def compare_with_benchmark(
    entities: Dict[str, Any],
    industry: str,
    volume_tier: str,
    pricing_type: str,
    region: str,
    bench_df: pd.DataFrame,
) -> Dict[str, Any]:
    mg = (industry or "").strip()
    vt = (volume_tier or "").strip()
    pt = (pricing_type or "").strip()
    re = (region or "").strip()

    eff = compute_effective_rate(entities)
    er_raw = eff.get("effective_rate_raw")

    ...

    df = bench_df[
        (bench_df["INDUSTRY"].astype(str).str.strip() == mg)
        & (bench_df["VOLUME_TIER"].astype(str).str.strip() == vt)
        & (bench_df["PRICING_TYPE"].astype(str).str.strip() == pt)
        & (bench_df["REGION"].astype(str).str.strip() == re)
    ].copy()

    ...

    pick = df.iloc[0]
    bench_raw = _to_float(pick["BENCHMARK"]) if pd.notna(pick.get("BENCHMARK")) else None
    delta = (er_raw - bench_raw) if (er_raw is not None and bench_raw is not None) else None
```

This is the main comparison function. It calls `compute_effective_rate` to get the merchant's ER, then filters the benchmark DataFrame to the exact row matching all four cascade selections, picks the benchmark value from that row, and computes delta — how far above or below benchmark this merchant is running.

The DataFrame filter is a four-way AND mask. All four dimensions have to match simultaneously. If no row matches, we return a structured result with a clear `notes` message explaining what combination we looked for and couldn't find. That's important for debugging — if someone adds a new pricing type to the UI but forgets to add it to the S3 CSV, the response tells you exactly what failed.

The function always returns a full structured dictionary even in error cases. The keys are always present — `effective_rate_raw`, `benchmark_raw`, `delta_vs_benchmark_raw`, plus the four selection values and notes. That consistency means callers never have to do defensive key checks.

---

## `normalize_isv_to_bps` and `lookup_isv_bps`

```python
def normalize_isv_to_bps(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except Exception:
        return None
    return f if f > 1.0 else to_bps(f)


def lookup_isv_bps(
    isv_df: Optional[pd.DataFrame],
    isv_name: Optional[str],
) -> Optional[float]:
    ...
    matches = isv_df[
        isv_df["ISV Name"].astype(str).str.strip().str.lower() == name
    ]
    if matches.empty:
        return None
    return normalize_isv_to_bps(matches.iloc[0]["ISV Benchmark"])
```

These two helpers work together for ISV lookup. `lookup_isv_bps` does a case-insensitive search on the ISV name — `.str.lower()` on both sides — and then passes the raw value to `normalize_isv_to_bps`.

The normalization logic is clever here. We don't enforce a strict format on the ISV CSV. If the value is greater than 1.0, we assume it's already in basis points and return it as-is. If it's 1.0 or less, we assume it's a decimal and convert it to BPS. So the same CSV can have some ISVs listed as `200` and others as `0.02` and it'll handle both correctly. That's a pragmatic choice — it reduces the data maintenance burden on whoever manages that CSV.

---

## Closing

That's `tsg_benchmark.py` — the complete utility layer. Pure functions, defensive parsing, no side effects. Every function we just walked through gets called either by `benchmark_main_pipeline.py` or directly by the Streamlit UI. Next we'll go into `benchmark_main_pipeline.py`, which is the orchestrator that pulls all of these together into a single pipeline call.
