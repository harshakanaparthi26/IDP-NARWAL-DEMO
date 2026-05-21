# Script 01 — `main.py` (Benchmark Trigger Section Only)

---

## Opening

Alright, so `main.py` is the FastAPI backend — the single entry point for the entire IDP system. We've covered most of this file in previous sessions: the data pipeline, the embedding pipeline, the HITL routing. We're not going back over any of that. What we're focused on today is just one specific section of this file — the benchmark trigger that lives inside `/document/ingest`. This is the moment where `main.py` hands off to the benchmark pipeline.

---

## The Import

```python
from backend.src.services.benchmark_comparison_pipeline.benchmark_main_pipeline import (
    run_benchmark_main_pipeline
)
```

Right at the top of the file, alongside all the other pipeline imports, we're pulling in `run_benchmark_main_pipeline` from `benchmark_main_pipeline.py`. That's the orchestrator function we'll cover in the next script. This is the only benchmark-related import in `main.py` — everything else lives downstream.

---

## The `/document/ingest` Endpoint — Context Params

```python
@app.post("/document/ingest")
async def document_ingest(
    file: UploadFile = File(...),
    context: str = Form("{}")
):
```

The `/document/ingest` endpoint accepts two things — the PDF file itself, and a `context` JSON string. That context string is how the caller passes in all the pipeline configuration, including the benchmark parameters. So when a frontend or API client submits a merchant statement, they can include things like industry, tier, pricing type, region, and ISV name right in that context payload. We'll see exactly how those get used in a moment.

---

## The Five Pipeline Steps — Where Benchmark Sits

```python
"""
End-to-end ingestion:
    1. Textract / data pipeline
    2. Embeddings
    3. Retriever / extraction
    4. Evaluation (flags REVIEW / INCORRECT)
    5. Optional benchmarking
"""
```

The docstring lays it out clearly. Benchmarking is step five, and it's marked as optional. Steps one through four have to complete successfully before we even get here. The data pipeline extracts text from the PDF, embeddings vectorize it, the retriever pulls the relevant chunks, the LLM extracts the entities, and the evaluation judge flags anything suspicious. Only then do we have the `separate_entities` dict — the extracted merchant data — that the benchmark pipeline needs to compute the effective rate.

---

## The Benchmark Trigger

```python
# --------------------------- 5. BENCHMARK ---------------------------
benchmark_out = None

if all(
    ctx.get(k)
    for k in ["industry", "tier", "pricing_type", "region"]
):
    try:
        with metrics_mgr.step("benchmark_pipeline"):

            benchmark_out = run_benchmark_main_pipeline(
                entities=separate_entities,
                industry=ctx["industry"],
                volume_tier=ctx["tier"],
                pricing_type=ctx["pricing_type"],
                region=ctx["region"],
                isv_name=ctx.get("isv"),
                metrics=metrics_mgr
            )

    except Exception as e:
        error_step = "benchmark_pipeline"
        error_message = str(e)
        raise
```

This is the key section. The first thing to notice is `benchmark_out = None` — the benchmark is entirely optional. If it doesn't run, the response just returns `None` for that field and everything else still works fine. The pipeline doesn't fail if benchmarking is skipped.

The condition is `if all(ctx.get(k) for k in ["industry", "tier", "pricing_type", "region"])`. We need all four filter parameters to be present and truthy before we even attempt to run the benchmark. That's a deliberate guard — without those four values, we can't look up the right row in the benchmark table, so there's no point calling the pipeline at all. The caller opts in by providing those context parameters.

Notice that `isv_name` is fetched with `ctx.get("isv")` — no default, no fallback. So ISV is optional within the optional benchmark step. If it's there, we generate ISV-based quotes. If it's not, the pipeline runs without it.

The whole thing is wrapped in `metrics_mgr.step("benchmark_pipeline")`, which is the metrics tracking context manager used throughout this file. It records timing and success/failure for this step just like every other pipeline step.

We then pass `separate_entities` directly into `run_benchmark_main_pipeline`. Those are the extracted entities from the retriever — specifically the `total_fees` and `total_amount` values that the benchmark pipeline uses to compute the merchant's effective rate. That's the handoff point. `main.py` does no benchmark math itself — it just assembles the inputs and delegates.

---

## The Final Response

```python
return {
    "ok": True,
    "doc_id": doc_id,
    "doc_name": file.filename,
    "s3_raw_key": s3_key,
    "bucket": settings.S3_BUCKET,
    "tables": tables,
    "extracted": separate_entities,
    "evaluation": evaluation_out.get("evaluation"),
    "evaluation_score": evaluation_out.get("evaluation_score"),
    "evaluation_flags": evaluation_out.get("evaluation_flags"),
    "benchmark": benchmark_out,
}
```

The benchmark output comes back as `benchmark_out` and gets included in the final response under the `"benchmark"` key. If the benchmark ran, that field contains the full structured result from `benchmark_main_pipeline.py`. If it didn't run, it's `None`. Simple, clean, and the rest of the response is completely unaffected either way.

---

## Closing

That's the entire benchmark-relevant section of `main.py`. It's intentionally thin — just a guard condition, a delegation call, and a response key. All the real logic lives downstream. Next we'll go into `tsg_benchmark.py`, which is the core utility layer that every other benchmark file depends on.
