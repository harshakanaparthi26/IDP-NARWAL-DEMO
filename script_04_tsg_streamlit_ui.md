# Script 04 — `_TSG_Benchmark_Comparission.py` (Streamlit UI)

---

## Opening

Last file. This is `_TSG_Benchmark_Comparission.py` — the Streamlit frontend for the benchmark step. In the UI flow, this is Step 4. After the document has been ingested, extracted, evaluated, and reviewed in HITL, the reviewer lands on this page to see the benchmark comparison and the generated quotes. Unlike the pipeline files we've covered, this one runs interactively in a browser, and it calls into `tsg_benchmark.py` directly — not through `benchmark_main_pipeline.py`. It's its own independent consumer of the utility layer.

---

## Imports

```python
from backend.src.storage.dynamodb_all import get_extraction

from backend.src.services.benchmark_comparison_pipeline.tsg_benchmark import (
    load_benchmarks,
    get_industries,
    get_volume_tiers,
    get_pricing_types,
    get_regions,
    compare_with_benchmark,
    to_bps,
    make_isv_quotes,
    load_isv_table,
    lookup_isv_bps,
)
```

The DynamoDB import — `get_extraction` — is covered by another part of the team. We're not going into it here. What matters for us is what comes after: everything imported from `tsg_benchmark.py`. You can see almost the entire public API of that utility file being imported here. The UI is a direct consumer of the same functions the pipeline uses — same loaders, same dropdown helpers, same comparison logic.

---

## Page Setup and Doc Loading

```python
st.set_page_config(
    page_title="TSG Benchmark Comparison",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    "<div class='step-header'><h3>Step 4 — TSG Benchmark Comparison</h3></div>",
    unsafe_allow_html=True,
)
```

```python
doc_id = ss.get("doc_id")

if not doc_id:
    st.info("No processed document found.")
    st.stop()

record = get_extraction(doc_id)
entities = record.get("separate_prompt_entities", {}) or {}
```

`ss` is the session state wrapper — it gives us a clean interface to Streamlit's session state dictionary. We use it throughout this file to store and retrieve user selections and loaded DataFrames across reruns.

The `doc_id` check is a gate. If no document has been processed yet, we show an info message and call `st.stop()`, which halts Streamlit execution immediately. That's the Streamlit-idiomatic way to do early exits — you don't use `return` in a Streamlit script the way you would in a function.

`record` is the full DynamoDB record for the processed document. We pull `separate_prompt_entities` from it — that's the extracted merchant data from the LLM, the same `entities` dict the pipeline uses to compute the effective rate.

---

## Benchmark Table Loading with Session State Cache

```python
if ss.get("bench_df") is None:
    df = load_benchmarks()

    if df is None:
        st.error("Benchmark table could not be loaded.")
        st.stop()

    ss.bench_df = df

bench_df = ss.bench_df
```

This is an important pattern in Streamlit. Streamlit reruns the entire script from top to bottom on every user interaction. If we called `load_benchmarks()` unconditionally, we'd hit S3 on every single dropdown change. Instead, we cache the DataFrame in session state. The first time the page loads, `ss.bench_df` is None so we fetch from S3 and store it. Every subsequent rerun, we skip the fetch and use the cached DataFrame directly. That's the right way to handle expensive loads in Streamlit.

---

## Cascading Sidebar Dropdowns

```python
industry_options = get_industries(bench_df)
ss.bench_industry = st.sidebar.selectbox("Industry", industry_options, index=0)

volume_options = get_volume_tiers(bench_df, ss.bench_industry)
ss.bench_tier = st.sidebar.selectbox("Volume Tier", volume_options, index=0)

pricing_options = get_pricing_types(bench_df, ss.bench_industry, ss.bench_tier)
ss.bench_pricing_type = st.sidebar.selectbox("Pricing Type", pricing_options, index=0)

region_options = get_regions(bench_df, ss.bench_industry, ss.bench_tier, ss.bench_pricing_type)
ss.bench_region = st.sidebar.selectbox("Region", region_options, index=0)
```

This is where the dropdown helper functions from `tsg_benchmark.py` get used directly. Each `selectbox` call feeds the selected value from the previous dropdown into the next helper function. So when you select an industry, `get_volume_tiers` filters down to only the tiers that exist for that industry. When you select a tier, `get_pricing_types` filters further. Each selection narrows the options for the next level.

Each selected value gets stored in session state — `ss.bench_industry`, `ss.bench_tier`, and so on. That's because Streamlit reruns the script on each interaction, and we need the previously selected values to persist so the downstream dropdowns initialize correctly.

---

## ISV Table and Dropdown

```python
isv_df = load_isv_table(
    "/home/ec2-user/SageMaker/IDP_MAIN_APPLICATION/ISVdetails.csv"
)

if isv_df is None:
    st.sidebar.error("❌ ISV table not loaded")
    st.stop()

isv_options = sorted(isv_df["ISV Name"].dropna().unique())

ss.bench_isv = st.sidebar.selectbox("ISV", isv_options, index=0)
```

Notice that unlike the benchmark table, the ISV table is not cached in session state. It's loaded on every rerun. That's a minor inefficiency — it reads from disk on every user interaction. If someone wanted to optimize this in a future pass, wrapping it in the same `if ss.get("isv_df") is None` pattern would be the right fix.

The path is passed explicitly here as a string literal — you can see the comment in the original code flagging it as something to update if the path changes. That's a known maintenance point.

---

## Running the Comparison

```python
comp = compare_with_benchmark(
    entities=entities,
    industry=ss.bench_industry,
    volume_tier=ss.bench_tier,
    pricing_type=ss.bench_pricing_type,
    region=ss.bench_region,
    bench_df=bench_df,
)

if not comp or comp.get("benchmark_raw") is None:
    st.warning("No benchmark available.")
    st.stop()
```

Every time the user changes a dropdown, Streamlit reruns the script and this `compare_with_benchmark` call fires again with the new selections. The result is always fresh. If no matching benchmark row is found — `benchmark_raw` is None — we show a warning and stop. The user needs to select a valid combination to get results.

---

## Computing BPS Values and ISV Quotes

```python
er = comp.get("effective_rate_raw")
bench = comp.get("benchmark_raw")
delta = comp.get("delta_vs_benchmark_raw")

er_bps = to_bps(er)

isv_name = ss.bench_isv
isv_bps = lookup_isv_bps(isv_df, isv_name)

quotes = {}

if er_bps is not None and isv_bps is not None:
    quotes = make_isv_quotes(er_bps, isv_bps)
```

We pull the three raw decimal values out of the comparison result, convert the effective rate to BPS, then use `lookup_isv_bps` from `tsg_benchmark.py` to get the ISV's benchmark in BPS. If both are available, we call `make_isv_quotes` to generate the three-tier quote. This is the clean path — using `lookup_isv_bps` rather than doing the inline lookup that `benchmark_main_pipeline.py` does. Same result, cleaner call.

---

## Displaying Metrics

```python
c1, c2, c3, c4 = st.columns(4)

c1.metric("Effective Rate (BPS)", f"{er_bps:.2f}" if er_bps else "—")
c2.metric("Benchmark", f"{bench:.6f}" if bench else "—")
c3.metric("Δ vs Benchmark", f"{delta:.6f}" if delta else "—")
c4.metric("ISV (BPS)", f"{isv_bps:.2f}" if isv_bps else "—")
```

Four metric cards across the top. Effective rate and ISV are shown in BPS. The benchmark and delta are shown in decimal to six decimal places — that's because those raw decimal values are small numbers like `0.024500` and showing them in BPS would be fine too, but this preserves the exact precision from the source data.

The `"—"` fallback on each metric is a UX decision — blank or zero would be confusing, a dash is the standard way to signal "not available" in a metrics display.

---

## Displaying Quotes

```python
if not quotes:
    st.warning("No quotes generated")

else:
    st.subheader("Recommended Quotes (BPS)")

    q1, q2, q3 = st.columns(3)

    q1.metric("Low Profitability", quotes.get("Low Profitability"))
    q2.metric("Mid Profitability", quotes.get("Mid Profitability"))
    q3.metric("High Profitability", quotes.get("High Profitability"))
```

Three metric cards for the three quote tiers — Low, Mid, and High Profitability. These are the values a sales or pricing team would actually use. The labels come directly from the dictionary keys returned by `make_isv_quotes` in `tsg_benchmark.py`.

---

## Debug and Details Expanders

```python
with st.expander("Debug Info"):
    st.write("Selected ISV:", isv_name)
    st.write("ER BPS:", er_bps)
    st.write("ISV BPS:", isv_bps)
    st.write("Quotes:", quotes)

with st.expander("Benchmark Details"):
    st.json(comp)
```

Two collapsible expanders at the bottom. The Debug Info expander shows the key computed values — useful during development and QA. The Benchmark Details expander renders the full `comp` dictionary as formatted JSON, which gives you the complete output of `compare_with_benchmark` including the notes field and all four selection values. This is what you'd look at if a comparison came back wrong and you needed to trace why.

---

## Closing

And that's the full benchmark pipeline — all four files. `main.py` triggers it after extraction and evaluation, `benchmark_main_pipeline.py` orchestrates it, `tsg_benchmark.py` does all the actual work, and the Streamlit UI gives reviewers an interactive view of the same comparison with live filters. The project is now documented end to end for anyone who picks it up in the future.
