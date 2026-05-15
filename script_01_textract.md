# SCRIPT 1 — textract.py
### "Document Ingestion via Amazon Textract"

---

## OPENING

"Alright, let's start with the first file in the data pipeline — `textract.py`. This is Step 1 of the entire system. Before we can do anything — extract fees, run PII redaction, generate quotes — we first need to get the raw text out of the merchant statement PDF. That's what this file does.

The file is responsible for three things: uploading the PDF to S3, running Amazon Textract to extract text and tables from it, and then saving those outputs back to S3. Let me walk through it top to bottom."

---

## IMPORTS

```python
import io
import time
import boto3
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import settings
```

"We import `boto3` — that's the AWS SDK for Python, which we use to talk to S3 and Textract. We import `pandas` because Textract returns table data as raw cell blocks, and we convert those into proper DataFrames for downstream use. `defaultdict` from `collections` is used to build those tables cleanly — I'll show you why in a moment.

One thing worth noting: we're not using any third-party OCR library like pytesseract or pdf2image here. We chose Textract specifically because it's designed for financial documents — it handles multi-column layouts, embedded tables, and handwriting much better than open-source alternatives. For merchant statements, which are dense and highly structured, that accuracy matters a lot."

---

## AWS CLIENTS

```python
_s3 = boto3.client("s3", region_name=settings.AWS_REGION)
_textract = boto3.client("textract", region_name=settings.AWS_REGION)
```

"We initialize two AWS clients at module load time — one for S3 and one for Textract. Both are prefixed with an underscore to signal they are private to this module. We pull the region from a shared `settings` module so it's configured in one place across the whole system."

---

## upload_to_s3

```python
def upload_to_s3(file: Any, filename: str) -> str:
    key = settings.S3_PREFIX_RAW + filename
    _s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=file.read(),
        ContentType="application/pdf",
    )
    print(f"[Textract] Uploaded -> s3://{settings.S3_BUCKET}/{key}")
    return key
```

"This is the first function the pipeline calls. It takes the raw file object and a filename, builds an S3 key using a prefix from settings — so all raw PDFs land in a consistent folder — and uploads the file using `put_object`. It sets the `ContentType` to `application/pdf` explicitly, which is good practice for S3 so the object is correctly typed. It returns the S3 key so downstream steps know where to find the file."

---

## _start_analysis

```python
def _start_analysis(bucket: str, key: str) -> str:
    resp = _textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "FORMS"],
    )
    return resp["JobId"]
```

"This private function kicks off the Textract async job. We call `start_document_analysis` — not `detect_document_text` — and there's an important reason for that. `detect_document_text` only extracts plain text. `start_document_analysis` with `FeatureTypes=['TABLES', 'FORMS']` also extracts structured table data, which is critical for us because merchant statements contain tables of interchange fees and transaction summaries. We return the `JobId` which we'll use to track the job."

---

## _poll_job

```python
def _poll_job(job_id: str, delay: float = 2.0, max_tries: int = 600) -> Dict:
    for _ in range(max_tries):
        resp = _textract.get_document_analysis(JobId=job_id)
        status = resp.get("JobStatus")

        if status == "SUCCEEDED":
            return resp

        if status in ("FAILED", "PARTIAL_SUCCESS"):
            raise RuntimeError(f"Textract job failed: {status}")

        time.sleep(delay)

    raise TimeoutError(f"Textract job {job_id} timed out.")
```

"Because Textract is async — it processes documents in the background — we need to poll for the result. This function loops up to 600 times with a 2-second delay between each check, which gives us a maximum wait time of 20 minutes. That's more than enough for even large merchant statement PDFs.

We explicitly handle three states: `SUCCEEDED` means we're good and return the response. `FAILED` or `PARTIAL_SUCCESS` means something went wrong and we raise immediately. If we exceed our retries, we raise a `TimeoutError`. We don't silently continue — we want any failure here to surface loudly."

---

## _paginate

```python
def _paginate(first_page: Dict, job_id: str) -> List[Dict]:
    pages = [first_page]
    token = first_page.get("NextToken")

    while token:
        resp = _textract.get_document_analysis(JobId=job_id, NextToken=token)
        pages.append(resp)
        token = resp.get("NextToken")

    return pages
```

"Textract paginates its results — a single job can return thousands of blocks across multiple API response pages. This function handles that transparently. We start with the first page we already have from `_poll_job`, check for a `NextToken`, and keep fetching until there's no more. We collect all pages into a list and return them. This is important — if you don't paginate, you silently drop data from multi-page PDFs."

---

## run_textract

```python
def run_textract(bucket: str, key: str) -> List[Dict]:
    job_id = _start_analysis(bucket, key)
    print(f"[Textract] Job started: {job_id}")

    first = _poll_job(job_id)
    pages = _paginate(first, job_id)

    print(f"[Textract] Job complete: {len(pages)} page(s) DONE")
    return pages
```

"This is the public-facing function that ties together start, poll, and paginate. It's a clean three-step sequence. Start the job, wait for the first result, then paginate through the rest. The caller just gets back a list of all Textract response pages — they don't need to know anything about the async mechanics underneath."

---

## parse_blocks — Text Lines

```python
def parse_blocks(pages: List[Dict]) -> Tuple[str, List[pd.DataFrame]]:
    text_lines: List[str] = []
    tables: List[pd.DataFrame] = []

    for page in pages:
        blocks = page.get("Blocks", [])
        block_map = {b["Id"]: b for b in blocks}

        for b in blocks:
            if b["BlockType"] == "LINE":
                text_lines.append(b.get("Text", ""))
```

"This is the most important parsing function in the file. Textract doesn't return text as a single string — it returns a list of `Block` objects, each with a type like `LINE`, `WORD`, `CELL`, `TABLE`, etc. We first build a `block_map` — a dictionary keyed by block ID — so we can do fast lookups when resolving parent-child relationships.

For plain text, we just filter for `LINE` blocks and collect their text. `LINE` is the right level here — `WORD` would be too granular and lose spacing, and higher-level blocks like `PAGE` don't carry text directly."

---

## parse_blocks — Tables

```python
        table_cells: Dict = defaultdict(lambda: defaultdict(str))
        has_cells = False

        for b in blocks:
            if b["BlockType"] != "CELL":
                continue

            has_cells = True
            r, c = b["RowIndex"], b["ColumnIndex"]
            words = []

            for rel in b.get("Relationships", []) or []:
                if rel["Type"] != "CHILD":
                    continue

                for cid in rel["Ids"]:
                    child = block_map.get(cid)

                    if not child:
                        continue

                    if child["BlockType"] == "WORD":
                        words.append(child.get("Text", ""))
                    elif (
                        child["BlockType"] == "SELECTION_ELEMENT"
                        and child.get("SelectionStatus") == "SELECTED"
                    ):
                        words.append("[X]")

            table_cells[r][c] = " ".join(words).strip()
```

"For tables, Textract gives us `CELL` blocks, each with a `RowIndex` and `ColumnIndex`. Each cell has `Relationships` that point to its child `WORD` blocks. We use a `defaultdict` of `defaultdict` to build a 2D grid — `table_cells[row][col]` — without having to pre-allocate the matrix size.

We also handle `SELECTION_ELEMENT` blocks — those are checkboxes in forms. If one is checked, we insert `[X]` as the cell text. That handles edge cases in merchant statement forms that have checkbox fields."

---

## parse_blocks — DataFrame Assembly

```python
        if has_cells and table_cells:
            max_row = max(table_cells.keys())
            max_col = max(max(cols.keys()) for cols in table_cells.values())

            rows = [
                [table_cells[r].get(c, "") for c in range(1, max_col + 1)]
                for r in range(1, max_row + 1)
            ]

            tables.append(pd.DataFrame(rows))

    raw_text = "\n".join(t for t in text_lines if t)
    print(f"[Textract] Parsed: {len(raw_text):,} chars, {len(tables)} table(s)")

    return raw_text, tables
```

"Once we've built the grid, we find the max row and column, then build a proper 2D list of rows which we wrap in a `pd.DataFrame`. We use `.get(c, '')` with an empty string default to handle any sparse cells where Textract detected a cell boundary but found no text. Empty string is the right fallback — not None — because downstream CSV writes and string comparisons are safer with empty strings.

We filter out empty strings from `text_lines` before joining, so we don't end up with lots of blank lines in the final text. We return both the `raw_text` string and the list of DataFrames."

---

## _save_outputs

```python
def _save_outputs(filename: str, raw_text: str, tables: List[pd.DataFrame]) -> str:
    base = filename.replace(".pdf", "")

    text_key = settings.S3_PREFIX_OUT + base + ".txt"
    _s3.put_object(Bucket=settings.S3_BUCKET, Key=text_key, Body=raw_text.encode("utf-8"))

    for i, df in enumerate(tables, 1):
        csv_key = settings.S3_PREFIX_OUT + base + f"_table_{i}.csv"
        _s3.put_object(Bucket=settings.S3_BUCKET, Key=csv_key, Body=df.to_csv(index=False).encode("utf-8"))

    return text_key
```

"We save the raw text as a `.txt` file and each table as a numbered `.csv` file — so if a document has 3 tables, we get `_table_1.csv`, `_table_2.csv`, `_table_3.csv`. We encode everything as UTF-8 before uploading. The function returns the S3 key for the text file, which is passed downstream as a reference."

---

## process_document — Main Entrypoint

```python
def process_document(file: Any, filename: str, metrics=None) -> Dict[str, Any]:
    try:
        s3_raw_key = upload_to_s3(file, filename)
        pages = run_textract(settings.S3_BUCKET, s3_raw_key)
        raw_text, tables = parse_blocks(pages)
        s3_text_key = _save_outputs(filename, raw_text, tables)

        return {
            "status": "success",
            "text": raw_text,
            "tables": tables,
            "s3_raw_key": s3_raw_key,
            "s3_text_key": s3_text_key,
        }

    except Exception as e:
        print(f"[Textract] ERROR: {e}")
        return {"status": "error", "error": str(e)}
```

"This is the single function that `data_main_pipeline.py` calls. It orchestrates all the steps in sequence — upload, extract, parse, save — and returns a clean dictionary with a `status` field. If anything fails, we catch the exception and return a structured error dict instead of letting an uncaught exception bubble up. That lets the pipeline caller make a clean decision on how to handle the failure.

The `metrics` parameter is passed through to log API latencies to our metrics system — I'll cover that more when we talk about `main.py`.

So that's `textract.py` — a clean, self-contained document ingestion layer. It handles async job management, pagination, table parsing, and S3 persistence all in one place."

---
