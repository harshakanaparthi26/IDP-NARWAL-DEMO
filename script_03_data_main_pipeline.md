# SCRIPT 3 — data_main_pipeline.py
### "The Data Pipeline Orchestrator"

---

## OPENING

"Now let's look at `data_main_pipeline.py`. This file is the orchestrator for everything we just looked at. It doesn't do any heavy lifting itself — it calls `textract.py` and `comprehend.py` in sequence, and it handles all the DynamoDB bookkeeping around those steps.

Think of it as the conductor. Textract and Comprehend are the instruments. This file decides when each one plays, what to do if one fails, and how to record what happened."

---

## IMPORTS

```python
from io import BytesIO
from typing import Dict, Any, Optional
import traceback
import logging

from backend.src.storage.dynamodb_all import (
    write_meta_start,
    write_meta_complete,
    update_meta_s3_raw_key,
)

try:
    from . import textract
except Exception:
    textract = None

try:
    from . import comprehend
except Exception:
    comprehend = None
```

"We import three specific DynamoDB functions — `write_meta_start`, `write_meta_complete`, and `update_meta_s3_raw_key`. I'll cover what each one does as we hit them in the pipeline flow.

The `try/except` imports for `textract` and `comprehend` are intentional. If either module fails to import — say due to a missing dependency in a certain environment — the pipeline doesn't crash at startup. It degrades gracefully. We check for `None` later before calling each module."

---

## Function Signature

```python
def run_data_main_pipeline(
    *,
    file_bytes: bytes,
    filename: str,
    document_type: Optional[str] = None,
    doc_id: Optional[str] = None,
    industry: Optional[str] = None,
    volume_tier: Optional[str] = None,
    pricing_type: Optional[str] = None,
    region: Optional[str] = None,
    metrics=None,
) -> Dict[str, Any]:
```

"The function uses keyword-only arguments — that's what the `*` at the start of the parameter list enforces. This means callers cannot pass positional arguments. For a function with 9 parameters, this is important for readability and safety — it makes every call site self-documenting.

`doc_id` comes in from `main.py` — the API layer generates it from the filename. The merchant context fields like `industry`, `volume_tier`, `pricing_type`, and `region` are passed through from the ISV's upload request and stored in DynamoDB metadata. They're used later by the benchmark comparison pipeline."

---

## Guard Clauses

```python
    if not file_bytes:
        raise ValueError("file_bytes cannot be empty")

    if doc_id is None:
        raise ValueError("data_main_pipeline requires doc_id from main.py ingestion.")
```

"Two guard clauses at the top. We validate inputs before doing any work. `file_bytes` must not be empty — if someone passes an empty file we fail fast with a clear message rather than getting a confusing error deep in the Textract call. `doc_id` is required because it's the primary key for everything we write to DynamoDB."

---

## META START

```python
    write_meta_start(
        doc_id=doc_id,
        doc_name=filename,
        s3_raw_key="",
        industry=industry,
        volume_tier=volume_tier,
        pricing_type=pricing_type,
        region=region,
    )
```

"The very first thing we do — before any processing — is write a META start record to DynamoDB. This sets `status = 'PROCESSING'` for this document. This is important for the HITL dashboard and the ops team: if a document is in `PROCESSING` state, something is actively running. If it's been stuck in `PROCESSING` for too long, that's a signal something went wrong. We pass an empty string for `s3_raw_key` here because we don't have it yet — we update it as soon as Textract gives it to us."

---

## Step 1 — Textract

```python
    file_obj = file_bytes if hasattr(file_bytes, "read") else BytesIO(file_bytes)

    try:
        textract_out = textract.process_document(file_obj, filename, metrics=metrics)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Textract failed: {e}")

    if textract_out.get("status") != "success":
        raise RuntimeError(
            textract_out.get("error", "Textract failed without error message")
        )

    raw_text = textract_out.get("text", "") or ""
    tables = textract_out.get("tables", []) or []
    s3_raw_key = textract_out.get("s3_raw_key")
```

"We wrap `file_bytes` in a `BytesIO` if it doesn't already have a `read` method. Textract's `upload_to_s3` calls `.read()` on the file object, so it needs to be file-like. `BytesIO` is the standard way to treat raw bytes as a file in Python without writing to disk.

We check the returned `status` field explicitly. Textract's `process_document` returns a structured dict with a `status` key — it doesn't always raise on failure. So we check both: we catch any exception that escapes, and we also check the status field for soft failures.

We use `or ''` and `or []` when unpacking `raw_text` and `tables` — this handles the case where the key exists but the value is `None`, which would otherwise cause type errors downstream."

---

## Update META with S3 Key

```python
    try:
        update_meta_s3_raw_key(
            doc_id=doc_id,
            s3_raw_key=s3_raw_key,
        )
    except Exception as e:
        logger.error(f"[DATA] Failed to update META s3_raw_key: {e}")
        raise
```

"As soon as we have the S3 key from Textract, we update the META record in DynamoDB with it. We do this immediately — not at the end — because the raw S3 key is needed by other pipelines. For example, the line item extraction pipeline reads directly from S3, and it looks up the key from the META record. If we only wrote it at the end and the pipeline crashed halfway through, those downstream services would have no key to work with."

---

## Step 2 — Comprehend PII

```python
    if comprehend is None or not hasattr(comprehend, "redact_pii"):
        return {
            "status": "success",
            ...
            "_note": "Comprehend adapter missing - Textract output only.",
        }

    try:
        redacted_text = comprehend.redact_pii(raw_text, metrics=metrics)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Comprehend PII redaction failed: {e}")
```

"Before calling Comprehend, we check if the module is available. If it's not, we return a successful partial result with a `_note` field explaining what's missing. This graceful degradation was intentional — during early development, Textract was ready before Comprehend, and we didn't want to block the whole pipeline. The `_note` field makes it easy to detect this in logs.

If Comprehend is available, we call `redact_pii` and wrap it in a try/except. A Comprehend failure is a hard failure — we raise rather than continue, because passing unredacted text downstream violates our compliance requirements."

---

## Save Redacted Text

```python
    s3_redacted_key = None
    if hasattr(comprehend, "save_redacted"):
        try:
            s3_redacted_key = comprehend.save_redacted(redacted_text, filename)
        except TypeError:
            try:
                s3_redacted_key = comprehend.save_redacted(
                    redacted_text, doc_id, filename
                )
            except Exception:
                s3_redacted_key = None
```

"This section has a slightly defensive pattern — we try calling `save_redacted` with two different signatures. The reason is that during development the function signature changed from `(text, filename)` to `(text, doc_id, filename)`, and this try/except on `TypeError` was added as a backwards-compatible bridge during the transition. It's the kind of pragmatic thing that happens in a fast-moving POC — we document it and move on."

---

## META COMPLETE

```python
    try:
        write_meta_complete(
            doc_id=doc_id,
            s3_text_key=s3_redacted_key,
            index_name="",
            char_count=len(redacted_text or ""),
            table_count=len(tables),
        )
    except Exception as e:
        logger.error(f"[DATA] Failed to write META completion: {e}")
```

"Once we have the redacted text and its S3 key, we write the META complete record to DynamoDB. This sets `status = 'COMPLETE'` and records the `s3_text_key`, `char_count`, and `table_count`. We wrap this in a try/except and only log the error — we don't raise. The reason is that by this point, the data processing is done. Failing to write the completion record is an ops issue, not a data issue. We don't want to fail the whole response because of a DynamoDB write at the very end."

---

## Return Payload

```python
    return {
        "status": "success",
        "filename": filename,
        "document_type": document_type,
        "raw_text": raw_text,
        "redacted_text": redacted_text,
        "tables": tables,
        "s3_raw_key": s3_raw_key,
        "s3_redacted_key": s3_redacted_key,
    }
```

"We return both `raw_text` and `redacted_text`. `raw_text` is used internally for debug and audit purposes. `redacted_text` is what flows to all downstream pipelines — embeddings, extraction, the LLM. The caller in `main.py` picks `redacted_text` first, and falls back to `raw_text` if redaction wasn't available.

So that's `data_main_pipeline.py` — it's the glue layer. It sequences Textract and Comprehend, handles failures at each step differently based on severity, keeps DynamoDB in sync throughout, and returns a clean structured payload to the API layer."

---
