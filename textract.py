# textract.py
"""
Step 1 — Document ingestion via Amazon Textract.

Functions:
    upload_to_s3(file, filename)            → s3_key
    run_textract(bucket, key)               → list of page dicts
    parse_blocks(pages)                     → (raw_text, tables)
    process_document(file, filename)        → {"text", "tables", "s3_key"}
"""
import io
import time
import boto3
import pandas as pd
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import settings


# ── AWS clients ───────────────────────────────────────────────────────────────
_s3       = boto3.client("s3",       region_name=settings.AWS_REGION)
_textract = boto3.client("textract", region_name=settings.AWS_REGION)


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_to_s3(file: Any, filename: str) -> str:
    """Upload raw PDF bytes to S3 and return the S3 key."""
    key = settings.S3_PREFIX_RAW + filename
    _s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=file.read(),
        ContentType="application/pdf",
    )
    print(f"[Textract] Uploaded → s3://{settings.S3_BUCKET}/{key}")
    return key


# ── Textract async helpers ────────────────────────────────────────────────────

def _start_analysis(bucket: str, key: str) -> str:
    resp = _textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "FORMS"],
    )
    return resp["JobId"]


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


def _paginate(first_page: Dict, job_id: str) -> List[Dict]:
    pages = [first_page]
    token = first_page.get("NextToken")
    while token:
        resp = _textract.get_document_analysis(JobId=job_id, NextToken=token)
        pages.append(resp)
        token = resp.get("NextToken")
    return pages


def run_textract(bucket: str, key: str) -> List[Dict]:
    """Start async Textract job and return all result pages."""
    job_id = _start_analysis(bucket, key)
    print(f"[Textract] Job started: {job_id}")
    first = _poll_job(job_id)
    pages = _paginate(first, job_id)
    print(f"[Textract] Job complete: {len(pages)} page(s)")
    return pages


# ── Block parser ──────────────────────────────────────────────────────────────

def parse_blocks(pages: List[Dict]) -> Tuple[str, List[pd.DataFrame]]:
    """
    Convert Textract block output into:
      - raw_text  : all LINE blocks joined by newlines
      - tables    : list of DataFrames (one per detected table)
    """
    text_lines: List[str] = []
    tables: List[pd.DataFrame] = []

    for page in pages:
        blocks = page.get("Blocks", [])
        block_map = {b["Id"]: b for b in blocks}

        # ── Text lines ──
        for b in blocks:
            if b["BlockType"] == "LINE":
                text_lines.append(b.get("Text", ""))

        # ── Tables ──
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
                    elif child["BlockType"] == "SELECTION_ELEMENT" and child.get("SelectionStatus") == "SELECTED":
                        words.append("[X]")
            table_cells[r][c] = " ".join(words).strip()

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


# ── Save outputs to S3 ────────────────────────────────────────────────────────

def _save_outputs(filename: str, raw_text: str, tables: List[pd.DataFrame]) -> str:
    """Save raw text and tables to S3. Returns the S3 key for the text file."""
    base = filename.replace(".pdf", "")

    text_key = settings.S3_PREFIX_OUT + base + ".txt"
    _s3.put_object(Bucket=settings.S3_BUCKET, Key=text_key, Body=raw_text.encode("utf-8"))

    for i, df in enumerate(tables, 1):
        csv_key = settings.S3_PREFIX_OUT + base + f"_table_{i}.csv"
        _s3.put_object(Bucket=settings.S3_BUCKET, Key=csv_key, Body=df.to_csv(index=False).encode("utf-8"))

    print(f"[Textract] Saved text → {text_key}, {len(tables)} table(s) to S3")
    return text_key


# ── Main entrypoint ───────────────────────────────────────────────────────────

def process_document(file: Any, filename: str) -> Dict[str, Any]:
    """
    Full Textract pipeline for one document.

    Returns:
        {
          "status": "success" | "error",
          "text":   str,
          "tables": List[pd.DataFrame],
          "s3_raw_key": str,
          "s3_text_key": str,
          "error":  str | None,
        }
    """
    try:
        s3_raw_key = upload_to_s3(file, filename)
        pages      = run_textract(settings.S3_BUCKET, s3_raw_key)
        raw_text, tables = parse_blocks(pages)
        s3_text_key = _save_outputs(filename, raw_text, tables)
        return {
            "status":       "success",
            "text":         raw_text,
            "tables":       tables,
            "s3_raw_key":   s3_raw_key,
            "s3_text_key":  s3_text_key,
        }
    except Exception as e:
        print(f"[Textract] ERROR: {e}")
        return {"status": "error", "error": str(e)}
