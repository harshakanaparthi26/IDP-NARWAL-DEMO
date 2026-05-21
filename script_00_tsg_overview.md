# Script 00 — TSG Benchmark Pipeline: Connections Overview

---

## Opening

Before we dive into the individual files, let me give you the full picture first — where this pipeline sits, how all the pieces connect, and what we're about to walk through.

---

## What This Pipeline Does

So this is the TSG Benchmark Comparison pipeline. What it does at a high level is take the extracted data from a merchant's statement — things like total fees and total transaction volume — and compare that merchant's effective rate against industry benchmark data that we pull from S3. From that comparison, we generate a three-tiered quote: a low profitability quote, a mid profitability quote, and a high profitability quote. Those quotes are what a sales or pricing team would actually use when going back to a merchant.

There's also an ISV layer on top of that. ISV stands for Independent Software Vendor — these are technology partners that sit between us and the merchant. Each ISV has their own benchmark rate, and we factor that in when building the quotes.

---

## Where It Sits in the Larger System

Now, this pipeline does not run in isolation. Before it ever gets triggered, a few other things have to happen first. A merchant statement PDF comes in, it goes through the data pipeline — Textract extracts the raw text, Comprehend processes it, and it gets stored in DynamoDB. Then the embedding pipeline vectorizes that text, the retrieval pipeline fetches the most relevant chunks, and the LLM extracts the key entities — things like total fees, total volume, effective rate. After that, the evaluation pipeline runs a judge over those extracted values and flags anything that looks wrong or needs human review.

Only after all of that has completed does the benchmark pipeline get a chance to run. So this pipeline sits at the very end of the `/document/ingest` flow, and it only fires if the caller passes in the right context — specifically industry, volume tier, pricing type, and region.

---

## The Four Files We're Covering

Let me walk you through the four files in this pipeline and how they connect to each other.

```
                    ┌──────────────────────────────────────────┐
                    │  Previously covered pipelines            │
                    │  Data → Embedding → Retrieval →          │
                    │  Extraction → Evaluation                 │
                    └──────────────────────────────────────────┘
                                        │
                                        │  extracted entities + context params
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │              main.py                     │
                    │  /document/ingest endpoint               │
                    │  Checks for industry, tier,              │
                    │  pricing_type, region in context         │
                    │  → calls run_benchmark_main_pipeline()   │
                    └──────────────────────────────────────────┘
                                        │
                                        │  passes entities + filter params
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │        benchmark_main_pipeline.py        │
                    │  Orchestrates the full benchmark flow    │
                    │  Loads tables, computes ER, compares,    │
                    │  converts to BPS, builds ISV quotes      │
                    │  → calls tsg_benchmark.py utilities      │
                    └──────────────────────────────────────────┘
                                        │
                                        │  delegates all computation
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │           tsg_benchmark.py               │
                    │  Core utility layer — no side effects    │
                    │  Loads S3 benchmark CSV                  │
                    │  Loads local ISV CSV                     │
                    │  Dropdown helpers, ER calc, BPS,         │
                    │  benchmark comparison, quote building    │
                    └──────────────────────────────────────────┘
                                        ▲
                                        │  also called directly
                                        │
                    ┌──────────────────────────────────────────┐
                    │    _TSG_Benchmark_Comparission.py        │
                    │  Streamlit UI — Step 4 of the review     │
                    │  Sidebar filters → calls tsg_benchmark   │
                    │  Displays metrics + 3-tier quotes        │
                    └──────────────────────────────────────────┘
```

So the flow goes like this. `main.py` is the entry point — it's the FastAPI backend that receives the merchant statement, runs all the upstream pipelines, and then at the very end, if the right context parameters are present, it calls `run_benchmark_main_pipeline()`.

`benchmark_main_pipeline.py` is the orchestrator. It loads the data tables if they haven't been loaded yet, computes the effective rate, runs the benchmark comparison, converts everything to basis points, and if an ISV was specified, builds the three-tier quote. It does all of this by delegating to `tsg_benchmark.py`.

`tsg_benchmark.py` is the pure utility layer. This is where all the actual logic lives — loading CSVs from S3 and from local disk, the cascading dropdown helpers, the effective rate formula, the BPS conversion, the benchmark comparison, and the quote generation math. No side effects, no writes, just pure functions.

And then `_TSG_Benchmark_Comparission.py` is the Streamlit frontend — it's Step 4 in the human review UI. It calls `tsg_benchmark.py` directly, not through the pipeline. The user selects filters from the sidebar, and the UI runs the comparison and displays the metrics and quotes inline.

---

## The ISV Table

One more thing worth calling out before we start — the ISV table. You'll see a `load_isv_table()` call in a couple of these files. The table it loads looks like this:

| ISV Name | ISV Benchmark |
|----------|---------------|
| CLOVER   | 200           |
| SQUARE   | 150           |
| SKYTAB   | 220           |
| KORONA   | 280           |
| REZKU    | 180           |

Simple two-column CSV. ISV Name is the lookup key, ISV Benchmark is the rate in basis points. That number is what we use to calculate the gap between the merchant's effective rate and the ISV's target, and from that gap we derive the three quote tiers. We'll see exactly how that math works when we get to `tsg_benchmark.py`.

---

## Closing

That's the full picture. We have four files, a clean separation between the API entry point, the orchestration layer, the utility layer, and the Streamlit UI. Let's now go file by file, starting with `main.py` — specifically just the benchmark trigger section — and then work our way down through the stack.
