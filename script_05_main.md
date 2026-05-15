# SCRIPT 5 — main.py
### "The API Layer — Where the Data Pipeline Gets Called"

---

## OPENING

"Finally, `main.py`. This is the FastAPI backend — the entry point the frontend calls. Every pipeline in the system is triggered from here.

I'm only going to cover the parts of this file that directly connect to the data pipeline — meaning where `data_main_pipeline.py`, `textract.py`, `comprehend.py`, and `dynamodb_all.py` are involved. The embedding, retriever, evaluation, benchmark, and HITL sections are owned by other teammates and will be covered by them."

---

## IMPORTS — The data pipeline connection

```python
from backend.src.services.data_pipeline.data_main_pipeline import run_data_main_pipeline

from backend.src.storage.dynamodb_all import (
    get_extraction,
    get_evaluation,
    list_all_evaluations,
    list_all_drifts,
)
```

"Two imports matter for us. First, `run_data_main_pipeline` — this is the function we just walked through in detail. That one call internally triggers `textract.process_document`, then `comprehend.redact_pii`, and writes to DynamoDB via `dynamodb_all` at every step. From `main.py`'s perspective it's a single function call.

Second, `dynamodb_all` — `main.py` imports read functions from it directly for the evaluation and HITL routes. The write functions — `write_meta_start`, `update_meta_s3_raw_key`, `write_meta_complete` — are called inside `data_main_pipeline.py`, not here."

---

## /document/ingest — Where the data pipeline is triggered

```python
@app.post("/document/ingest")
async def document_ingest(
    file: UploadFile = File(...),
    context: str = Form("{}")
):
    file_bytes = await file.read()
    ctx = json.loads(context or "{}")
    doc_id = ctx.get("doc_id") or _sanitize_identifier(file.filename)

    metrics_mgr = MetricsManager(doc_id=doc_id, doc_name=file.filename)

    with metrics_mgr.step("data_pipeline"):
        data_out = run_data_main_pipeline(
            file_bytes=file_bytes,
            filename=file.filename,
            doc_id=doc_id,
            industry=ctx.get("industry"),
            volume_tier=ctx.get("tier"),
            pricing_type=ctx.get("pricing_type"),
            region=ctx.get("region"),
            metrics=metrics_mgr
        )
```

"This is where it all starts. The ISV uploads a PDF and this endpoint fires. We read the raw file bytes with `await file.read()`, parse the context JSON, and generate the `doc_id` — either from the caller or from the filename via `_sanitize_identifier`.

Then we call `run_data_main_pipeline`. This is the handoff point into everything we covered in the previous scripts. Inside that call, `textract.py` uploads the PDF and extracts the text, `comprehend.py` redacts PII, and `dynamodb_all.py` writes the META record at the start and end. All of that happens inside this one call."

---

## Extracting the data pipeline output

```python
    raw_text = (
        data_out.get("redacted_text")
        or data_out.get("raw_text")
        or ""
    )

    s3_key = data_out.get("s3_raw_key")

    if not s3_key:
        raise RuntimeError("Ingestion failed: S3 key missing from data pipeline output")
```

"Once `run_data_main_pipeline` returns, we pull out what we need. We prefer `redacted_text` — that's the Comprehend output. If Comprehend wasn't available and only raw text came back, we fall back to that.

We also hard-guard on `s3_raw_key`. If the data pipeline didn't return an S3 key — meaning Textract didn't complete successfully — we raise immediately. Everything downstream depends on that key existing."

---

## CLOSING

"So from `main.py`'s perspective, the entire data pipeline — Textract, Comprehend, all three DynamoDB writes — is one function call: `run_data_main_pipeline`. The output of that call feeds the rest of the pipeline chain. That's the connection point between `main.py` and everything we covered in the earlier scripts."

---
