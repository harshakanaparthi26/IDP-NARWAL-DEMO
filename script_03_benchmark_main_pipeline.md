# Script 03 — `benchmark_main_pipeline.py`

---

## Opening

Now we're in `benchmark_main_pipeline.py`. This is the orchestrator — the file that `main.py` actually calls when it triggers the benchmark pipeline. If `tsg_benchmark.py` is the toolkit, this file is the one that picks up the tools and runs them in the right order. It's also the layer that produces the final structured output that goes back to `main.py` and into the API response.

This file is where `main.py` hands off to — when you saw `run_benchmark_main_pipeline()` being called in the ingest endpoint, this is the function that runs.

---

## Imports

```python
from . import (
    load_benchmarks,
    load_isv_table,
    compute_effective_rate,
    compare_with_benchmark,
    to_bps,
    make_isv_quotes,
)
```

All imports come from the package's `__init__.py` — that's what the relative import `.` refers to. All of these functions live in `tsg_benchmark.py`. The `__init__` re-exports them so this orchestrator doesn't need to know which specific module inside the package each function comes from. That's a clean package design — the internals are free to be reorganized without changing this file's imports.

---

## Function Signature

```python
def run_benchmark_main_pipeline(
    *,
    entities: Dict[str, Any],
    industry: str,
    volume_tier: str,
    pricing_type: str,
    region: str,
    isv_name: Optional[str] = None,
    bench_df: Optional[pd.DataFrame] = None,
    isv_df: Optional[pd.DataFrame] = None,
    metrics=None,
) -> Dict[str, Any]:
```

A few things worth pointing out about this signature. It's keyword-only — that `*` at the start means every argument must be passed by name. That's intentional for a function with this many parameters. It prevents silent argument order bugs.

`bench_df` and `isv_df` are both optional DataFrames. This is the same dependency injection pattern we saw in `tsg_benchmark.py`. If you already have the DataFrames loaded, you can pass them in and skip the S3 and file reads. If you don't, the function loads them itself. That means this pipeline can run standalone, but it also means the Streamlit UI can pre-load the tables and reuse them across multiple calls without hitting S3 every time.

`isv_name` is optional because ISV-based quoting is an optional feature. `metrics` is also optional — it's the `MetricsManager` instance from `main.py`, used only when called from the API. When called in other contexts, it's not needed.

---

## Table Loading

```python
bench_df = bench_df if bench_df is not None else load_benchmarks()

if bench_df is None or bench_df.empty:
    return {"ok": False, "message": "Benchmark table unavailable."}
```

First thing we do is ensure we have the benchmark table. If it wasn't passed in, we call `load_benchmarks()` from `tsg_benchmark.py` to fetch it from S3. If that returns None or an empty DataFrame — meaning S3 was unreachable or the CSV was malformed — we return an early failure response immediately. There's no point continuing without the benchmark table.

---

## Effective Rate

```python
eff = compute_effective_rate(entities or {})
er_raw = eff.get("effective_rate_raw")
er_bps = to_bps(er_raw)
```

We call `compute_effective_rate` from `tsg_benchmark.py` on the entities dict. That gives us the merchant's effective rate as a decimal. Then we immediately convert it to BPS with `to_bps`. We keep both — `er_raw` for the benchmark comparison math, and `er_bps` for the ISV quote math and for the output. The `entities or {}` guard means if entities is None we safely pass an empty dict rather than crashing inside `compute_effective_rate`.

---

## Benchmark Comparison

```python
comp = compare_with_benchmark(
    entities=entities or {},
    industry=industry or "",
    volume_tier=volume_tier or "",
    pricing_type=pricing_type or "",
    region=region or "",
    bench_df=bench_df,
)

bench_raw = comp.get("benchmark_raw")
delta_raw = comp.get("delta_vs_benchmark_raw")

bench_bps = to_bps(bench_raw)
delta_bps = to_bps(delta_raw) if delta_raw is not None else None
```

We call `compare_with_benchmark` from `tsg_benchmark.py` with all four filter parameters. That function filters the benchmark DataFrame to the matching row and computes the delta between the merchant's ER and the benchmark. We then convert both the benchmark value and the delta to BPS for the output. Notice `delta_bps` has an extra None check — because delta is a derived value that depends on both ER and benchmark being present, it can be None if either is missing, and we don't want to pass None into `to_bps` unnecessarily.

---

## ISV Quotes — Optional Block

```python
quotes = {}

if isv_name:
    isv_df = isv_df if isv_df is not None else load_isv_table()

    if isv_df is not None and not isv_df.empty and er_bps is not None:

        row = isv_df[
            isv_df["ISV Name"]
            .astype(str)
            .str.strip()
            .str.lower()
            == str(isv_name).strip().lower()
        ]

        if not row.empty:
            val = row.iloc[0]["ISV Benchmark"]

            try:
                isv_bps = (
                    float(val)
                    if float(val) > 1.0
                    else to_bps(float(val))
                )

                if isv_bps is not None:
                    quotes = make_isv_quotes(
                        eff_bps=er_bps,
                        isv_bps=isv_bps
                    )

            except Exception:
                pass
```

This entire block only runs if `isv_name` was provided. We load the ISV table if needed, do a case-insensitive lookup for the ISV name, normalize the benchmark value to BPS using the same greater-than-one heuristic we saw in `tsg_benchmark.py`, and then call `make_isv_quotes` to generate the three-tier quote.

Worth noting: this block has its own inline ISV lookup logic, which you'll notice is slightly redundant with `lookup_isv_bps` in `tsg_benchmark.py`. The Streamlit UI uses the clean `lookup_isv_bps` helper. This orchestrator does the lookup inline. Both approaches produce the same result. That's a minor inconsistency that could be cleaned up in a future refactor — but it works correctly as-is.

The bare `except Exception: pass` at the bottom is intentional. ISV quoting is a soft feature. If anything goes wrong during the ISV lookup or quote generation, we silently fall through and return empty quotes rather than failing the entire pipeline. The core benchmark comparison result is still returned regardless.

---

## Return Value

```python
return {
    "ok": True,
    "message": "OK",
    "effective_rate": {
        "raw": er_raw,
        "bps": er_bps
    },
    "benchmark": {
        "raw": bench_raw,
        "bps": bench_bps
    },
    "delta_vs_benchmark": {
        "raw": delta_raw,
        "bps": delta_bps
    },
    "quotes_bps": quotes,
    "selections": {
        "industry": industry,
        "volume_tier": volume_tier,
        "pricing_type": pricing_type,
        "region": region,
    },
    "notes": (
        (eff.get("notes") or "")
        + (
            (" | " + (comp.get("notes") or ""))
            if comp.get("notes")
            else ""
        )
    ),
}
```

The return structure is clean and consistent. Every numeric value is returned at both levels — raw decimal and BPS — so the caller can use whichever they need. `selections` echoes back the four filter parameters that were used, which is useful for auditability. And `notes` concatenates the notes strings from both `compute_effective_rate` and `compare_with_benchmark`, pipe-delimited. So if the effective rate computation had an issue and the benchmark lookup had an issue, both are captured in a single string in the output. That's the whole audit trail rolled up into one field.

This is the exact object that ends up in the `"benchmark"` key of the `/document/ingest` response we saw in `main.py`.

---

## Closing

That's `benchmark_main_pipeline.py` — a clean orchestrator that loads, computes, compares, converts, and optionally quotes, then returns a structured result. It delegates all math to `tsg_benchmark.py` and handles the optional ISV layer on top. The last file we'll cover is the Streamlit UI — `_TSG_Benchmark_Comparission.py` — which is the human-facing interface for Step 4 of the review flow.
