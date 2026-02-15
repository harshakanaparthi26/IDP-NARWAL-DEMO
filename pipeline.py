# pipeline.py
"""
Main pipeline orchestrator.

All pipeline steps are called from here in order.
All print() statements that appear in the UI live here.

Functions:
    run_document_pipeline(file, filename, config)
        → PipelineResult  (dict with all outputs)

    ask(question, doc_state)
        → answer_dict

    compute_effective_rate(entities)
        → rate_dict

    compare_with_benchmark(entities, merchant_group, bench_df)
        → comparison_dict
"""
from __future__ import annotations

import hashlib
import time
import re
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import boto3
import pandas as pd

import settings
import textract
import comprehend
import rag
import bedrock


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_doc_id(filename: str) -> str:
    """Generate a short deterministic document ID."""
    base = f"{filename}::{time.time()}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# Effective rate + benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float(val) -> Optional[float]:
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


def compute_effective_rate(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute effective rate = abs(total_fees) / abs(total_amount).
    Returns a dict with 'effective_rate_raw' and 'notes'.
    """
    ta = _to_float(entities.get("total_amount"))
    tf = _to_float(entities.get("total_fees"))
    notes = []
    er = None

    if ta is None:
        notes.append("total_amount missing")
    elif ta == 0:
        notes.append("total_amount is 0 — division prevented")
    if tf is None:
        notes.append("total_fees missing")

    if ta and ta != 0 and tf is not None:
        er = abs(tf) / abs(ta)

    return {
        "effective_rate_raw": er,
        "notes": " | ".join(notes),
    }


def load_benchmarks() -> Optional[pd.DataFrame]:
    """Load TSG benchmark CSV from S3. Returns DataFrame or None on failure."""
    try:
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=settings.S3_SNOWFLAKE_BUCKET, Key=settings.TSG_CSV_KEY)
        df  = pd.read_csv(BytesIO(obj["Body"].read()))
        df["MERCHANT_GROUP"]  = df["MERCHANT_GROUP"].astype(str).str.strip()
        df["VOLUME_TIER_TSG"] = df["VOLUME_TIER_TSG"].astype(str).str.strip()
        df["BENCHMARK"]       = pd.to_numeric(df["BENCHMARK"], errors="coerce")
        print(f"[Pipeline] Loaded {len(df):,} benchmark rows from S3")
        return df
    except Exception as e:
        print(f"[Pipeline] Could not load benchmarks: {e}")
        return None


def compare_with_benchmark(
    entities: Dict[str, Any],
    merchant_group: str,
    bench_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare effective rate vs benchmark for the selected merchant industry.
    Returns delta_vs_benchmark_raw and matched tier info.
    """
    eff = compute_effective_rate(entities)
    er  = eff.get("effective_rate_raw")

    mg = (merchant_group or "").strip()
    df_mg = bench_df[bench_df["MERCHANT_GROUP"] == mg]

    if df_mg.empty:
        return {"benchmark_raw": None, "delta_vs_benchmark_raw": None,
                "merchant_group": mg, "volume_tier_id": None, "volume_tier_tsg": None,
                "notes": f"No benchmark rows for '{mg}'"}

    # Pick latest process month + load date if available
    for col in ("PROCESS_MONTH", "LOAD_DATE"):
        if col in df_mg.columns:
            df_mg = df_mg[df_mg[col] == df_mg[col].max()]

    pick      = df_mg.iloc[0]
    bench_raw = float(pick["BENCHMARK"]) if pd.notna(pick.get("BENCHMARK")) else None
    delta     = (er - bench_raw) if (er is not None and bench_raw is not None) else None

    return {
        "effective_rate_raw":      er,
        "benchmark_raw":           bench_raw,
        "delta_vs_benchmark_raw":  delta,
        "merchant_group":          mg,
        "volume_tier_id":          int(pick["VOLUME_TIER_ID"]) if pd.notna(pick.get("VOLUME_TIER_ID")) else None,
        "volume_tier_tsg":         str(pick["VOLUME_TIER_TSG"]) if pd.notna(pick.get("VOLUME_TIER_TSG")) else None,
        "notes":                   "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_document_pipeline(
    file: Any,
    filename: str,
    bench_df: Optional[pd.DataFrame] = None,
    merchant_group: Optional[str]    = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Full IDP pipeline for one document.

    Steps:
      1. Textract  — upload + OCR + parse
      2. Comprehend — PII redaction
      3. RAG        — chunk + embed + index into OpenSearch
      4. Bedrock    — entity extraction (single prompt AND separate prompts)
      5. Derived    — effective rate + benchmark compare

    Args:
        file:            file-like object (Streamlit UploadedFile)
        filename:        original file name
        bench_df:        TSG benchmark DataFrame (optional)
        merchant_group:  selected MERCHANT_GROUP for benchmark (optional)
        on_status:       callback(message: str) for UI status updates

    Returns a flat dict with all outputs. Keys:
        doc_id, doc_name, s3_text_key, tables,
        raw_text (not persisted to S3),
        single_prompt_entities, separate_prompt_entities,
        effective_rate_raw, benchmark_comparison,
        index_name,
        error (None on success)
    """
    def status(msg: str) -> None:
        print(msg)
        if on_status:
            on_status(msg)

    doc_id = make_doc_id(filename)

    # ── Step 1: Textract ──────────────────────────────────────────────────────
    status(f"[1/4] Extracting text with Amazon Textract: {filename}")
    textract_out = textract.process_document(file, filename)

    if textract_out.get("status") != "success":
        return {"error": textract_out.get("error"), "doc_id": doc_id, "doc_name": filename}

    raw_text = textract_out["text"]
    tables   = textract_out["tables"]
    print(f"[Pipeline] Textract done: {len(raw_text):,} chars, {len(tables)} table(s)")

    # ── Step 2: PII redaction ─────────────────────────────────────────────────
    status(f"[2/4] Redacting PII with Amazon Comprehend: {filename}")
    redacted_text  = comprehend.redact_pii(raw_text)
    s3_text_key    = comprehend.save_redacted(redacted_text, doc_id, filename)

    # ── Step 3: RAG index ─────────────────────────────────────────────────────
    status(f"[3/4] Building RAG index in OpenSearch: {filename}")
    index_name = rag.build_index(redacted_text, doc_id=doc_id, doc_name=filename)

    # Convenience wrapper so bedrock.py never imports rag.py
    def retrieve_fn(query: str) -> List[Dict]:
        return rag.retrieve(query, doc_id=doc_id, index_name=index_name)

    # ── Step 4a: Entity extraction — single prompt ────────────────────────────
    status(f"[4/4] Extracting entities (single prompt): {filename}")
    general_chunks        = retrieve_fn("total amount total fees total transactions")
    single_entities       = bedrock.extract_entities_single_prompt(general_chunks)

    # ── Step 4b: Entity extraction — separate prompts ─────────────────────────
    status(f"[4/4] Extracting entities (separate prompts): {filename}")
    separate_entities     = bedrock.extract_entities_separate_prompts(retrieve_fn)

    # ── Step 5: Derived — effective rate + benchmark ──────────────────────────
    eff_rate              = compute_effective_rate(separate_entities)
    benchmark_comparison  = None

    if bench_df is not None and merchant_group:
        benchmark_comparison = compare_with_benchmark(separate_entities, merchant_group, bench_df)
        print(f"[Pipeline] Benchmark delta: {benchmark_comparison.get('delta_vs_benchmark_raw')}")

    print(f"[Pipeline] ✓ Complete: {filename} (doc_id={doc_id})")

    return {
        "error":                  None,
        "doc_id":                 doc_id,
        "doc_name":               filename,
        "s3_text_key":            s3_text_key,
        "tables":                 tables,
        "index_name":             index_name,
        # Extraction results (both methods)
        "single_prompt_entities":   single_entities,
        "separate_prompt_entities": separate_entities,
        # Derived
        "effective_rate_raw":     eff_rate.get("effective_rate_raw"),
        "effective_rate_notes":   eff_rate.get("notes", ""),
        "benchmark_comparison":   benchmark_comparison,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A (called live from the UI)
# ═══════════════════════════════════════════════════════════════════════════════

def ask(question: str, doc_id: str, index_name: str) -> Dict[str, Any]:
    """
    Answer a question about an already-indexed document.

    Args:
        question:    user's question
        doc_id:      document identifier
        index_name:  OpenSearch index to query

    Returns:
        {
          "answer":       str,
          "chunks_used":  int,
          "top_k_chunks": list,
          "judge_result": dict,
          "cache_hit":    bool,
        }
    """
    print(f"[Pipeline] Q&A: {question[:80]}")
    chunks = rag.retrieve(question, doc_id=doc_id, index_name=index_name)
    result = bedrock.ask_question(question, chunks)
    print(f"[Pipeline] Answer ready (chunks_used={result['chunks_used']})")
    return result
