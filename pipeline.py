# pipeline.py
"""
Main pipeline orchestrator.

All pipeline steps are called from here in order.
All print() statements that appear in the UI live here.

Functions:
    run_document_pipeline(file, filename, bench_df, merchant_group, volume_tier_tsg, on_status)
    ask(question, doc_id, index_name)
    compute_effective_rate(entities)
    get_industries(bench_df)
    get_volume_tiers(bench_df, merchant_group)
    compare_with_benchmark(entities, merchant_group, volume_tier_tsg, bench_df)
    load_benchmarks()
"""
from __future__ import annotations

import hashlib
import re
import time
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
    base = f"{filename}::{time.time()}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


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


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_benchmarks() -> Optional[pd.DataFrame]:
    """Load TSG benchmark CSV from S3. Returns cleaned DataFrame or None."""
    try:
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=settings.S3_SNOWFLAKE_BUCKET, Key=settings.TSG_CSV_KEY)
        df  = pd.read_csv(BytesIO(obj["Body"].read()))

        for col in ["VOLUME_TIER_ID", "VOLUME_TIER_TSG", "MERCHANT_GROUP", "BENCHMARK"]:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        df["VOLUME_TIER_ID"]  = pd.to_numeric(df["VOLUME_TIER_ID"], errors="coerce").astype("Int64")
        df["MERCHANT_GROUP"]  = df["MERCHANT_GROUP"].astype(str).str.strip()
        df["VOLUME_TIER_TSG"] = df["VOLUME_TIER_TSG"].astype(str).str.strip()
        df["BENCHMARK"]       = pd.to_numeric(df["BENCHMARK"], errors="coerce")

        print(f"[Pipeline] Loaded {len(df):,} benchmark rows from S3")
        return df
    except Exception as e:
        print(f"[Pipeline] Could not load benchmarks: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Dropdown data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_industries(bench_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique MERCHANT_GROUP values."""
    return sorted(bench_df["MERCHANT_GROUP"].dropna().astype(str).str.strip().unique().tolist())


def get_volume_tiers(bench_df: pd.DataFrame, merchant_group: str) -> List[str]:
    """
    Return volume tier labels (VOLUME_TIER_TSG) available for the selected industry.
    Filters to the latest PROCESS_MONTH so only current tiers appear.
    Sorted by VOLUME_TIER_ID for correct numeric order.
    """
    mg = (merchant_group or "").strip()
    if not mg:
        return []

    df = bench_df[bench_df["MERCHANT_GROUP"] == mg].copy()
    if df.empty:
        return []

    if "PROCESS_MONTH" in df.columns:
        df = df[df["PROCESS_MONTH"] == df["PROCESS_MONTH"].max()]

    if "VOLUME_TIER_ID" in df.columns:
        df = df.drop_duplicates("VOLUME_TIER_TSG").sort_values("VOLUME_TIER_ID")
        return df["VOLUME_TIER_TSG"].dropna().astype(str).str.strip().tolist()

    return sorted(df["VOLUME_TIER_TSG"].dropna().astype(str).str.strip().unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# Effective rate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_rate(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    effective_rate_raw = abs(total_fees) / abs(total_amount)
    Returns {"effective_rate_raw": float | None, "notes": str}
    """
    ta = _to_float(entities.get("total_amount"))
    tf = _to_float(entities.get("total_fees"))
    notes, er = [], None

    if ta is None:
        notes.append("total_amount missing")
    elif ta == 0:
        notes.append("total_amount is 0 — division prevented")
    if tf is None:
        notes.append("total_fees missing")
    if ta and ta != 0 and tf is not None:
        er = abs(tf) / abs(ta)

    return {"effective_rate_raw": er, "notes": " | ".join(notes)}


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark comparison (requires BOTH industry + volume tier)
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_benchmark(
    entities:        Dict[str, Any],
    merchant_group:  str,
    volume_tier_tsg: str,
    bench_df:        pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare effective rate vs benchmark for the selected industry + volume tier.

    Looks up the latest row matching (MERCHANT_GROUP, VOLUME_TIER_TSG),
    using PROCESS_MONTH and LOAD_DATE to get the most recent benchmark.

    Returns:
        {
            "effective_rate_raw":     float | None,
            "benchmark_raw":          float | None,
            "delta_vs_benchmark_raw": float | None,   # effective - benchmark
            "merchant_group":         str,
            "volume_tier_id":         int | None,
            "volume_tier_tsg":        str,
            "notes":                  str,
        }
    """
    mg = (merchant_group  or "").strip()
    vt = (volume_tier_tsg or "").strip()

    eff    = compute_effective_rate(entities)
    er_raw = eff.get("effective_rate_raw")

    if not mg or not vt:
        return {
            "effective_rate_raw": er_raw, "benchmark_raw": None,
            "delta_vs_benchmark_raw": None, "merchant_group": mg,
            "volume_tier_id": None, "volume_tier_tsg": vt,
            "notes": "Industry and/or volume tier not selected",
        }

    df = bench_df[
        (bench_df["MERCHANT_GROUP"] == mg) &
        (bench_df["VOLUME_TIER_TSG"] == vt)
    ].copy()

    if df.empty:
        return {
            "effective_rate_raw": er_raw, "benchmark_raw": None,
            "delta_vs_benchmark_raw": None, "merchant_group": mg,
            "volume_tier_id": None, "volume_tier_tsg": vt,
            "notes": f"No benchmark found for '{mg}' / '{vt}'",
        }

    # Latest process month → latest load date
    if "PROCESS_MONTH" in df.columns:
        df = df[df["PROCESS_MONTH"] == df["PROCESS_MONTH"].max()]
    if "LOAD_DATE" in df.columns:
        df = df[df["LOAD_DATE"] == df["LOAD_DATE"].max()]

    pick      = df.iloc[0]
    bench_raw = float(pick["BENCHMARK"]) if pd.notna(pick.get("BENCHMARK")) else None
    delta     = (er_raw - bench_raw) if (er_raw is not None and bench_raw is not None) else None

    print(f"[Pipeline] Benchmark lookup: '{mg}' / '{vt}' → benchmark={bench_raw}  delta={delta}")

    return {
        "effective_rate_raw":     er_raw,
        "benchmark_raw":          bench_raw,
        "delta_vs_benchmark_raw": delta,
        "merchant_group":         mg,
        "volume_tier_id":         int(pick["VOLUME_TIER_ID"]) if pd.notna(pick.get("VOLUME_TIER_ID")) else None,
        "volume_tier_tsg":        vt,
        "notes":                  eff.get("notes", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_document_pipeline(
    file:            Any,
    filename:        str,
    bench_df:        Optional[pd.DataFrame]          = None,
    merchant_group:  Optional[str]                   = None,
    volume_tier_tsg: Optional[str]                   = None,
    on_status:       Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Full IDP pipeline for one document.

    Steps:
      1. Textract   — upload + OCR + parse blocks → text + tables
      2. Comprehend — PII redaction → save to S3
      3. RAG        — chunk + embed + index into OpenSearch
      4a. Bedrock   — entity extraction (single prompt: all 3 at once)
      4b. Bedrock   — entity extraction (separate prompts: one per entity + judge)
      5.  Derived   — effective rate + benchmark comparison

    benchmark_comparison is only populated when both merchant_group
    AND volume_tier_tsg are provided.
    """
    def status(msg: str) -> None:
        print(msg)
        if on_status:
            on_status(msg)

    doc_id = make_doc_id(filename)

    # ── 1. Textract ───────────────────────────────────────────────────────────
    status(f"[1/4] Extracting text with Amazon Textract: {filename}")
    textract_out = textract.process_document(file, filename)

    if textract_out.get("status") != "success":
        return {"error": textract_out.get("error"), "doc_id": doc_id, "doc_name": filename}

    raw_text = textract_out["text"]
    tables   = textract_out["tables"]
    print(f"[Pipeline] Textract done: {len(raw_text):,} chars, {len(tables)} table(s)")

    # ── 2. PII redaction ──────────────────────────────────────────────────────
    status(f"[2/4] Redacting PII with Amazon Comprehend: {filename}")
    redacted_text = comprehend.redact_pii(raw_text)
    s3_text_key   = comprehend.save_redacted(redacted_text, doc_id, filename)

    # ── 3. RAG index ──────────────────────────────────────────────────────────
    status(f"[3/4] Building RAG index in OpenSearch: {filename}")
    index_name = rag.build_index(redacted_text, doc_id=doc_id, doc_name=filename)

    def retrieve_fn(query: str) -> List[Dict]:
        return rag.retrieve(query, doc_id=doc_id, index_name=index_name)

    # ── 4a. Single-prompt extraction ──────────────────────────────────────────
    status(f"[4/4] Extracting entities (single prompt): {filename}")
    general_chunks  = retrieve_fn("total amount total fees total transactions")
    single_entities = bedrock.extract_entities_single_prompt(general_chunks)

    # ── 4b. Separate-prompt extraction ────────────────────────────────────────
    status(f"[4/4] Extracting entities (separate prompts): {filename}")
    separate_entities = bedrock.extract_entities_separate_prompts(retrieve_fn)

    # ── 5. Effective rate + benchmark ─────────────────────────────────────────
    eff_rate = compute_effective_rate(separate_entities)

    benchmark_comparison = None
    if bench_df is not None and merchant_group and volume_tier_tsg:
        benchmark_comparison = compare_with_benchmark(
            entities        = separate_entities,
            merchant_group  = merchant_group,
            volume_tier_tsg = volume_tier_tsg,
            bench_df        = bench_df,
        )

    print(f"[Pipeline] ✓ Complete: {filename} (doc_id={doc_id})")

    return {
        "error":                    None,
        "doc_id":                   doc_id,
        "doc_name":                 filename,
        "s3_text_key":              s3_text_key,
        "tables":                   tables,
        "index_name":               index_name,
        "single_prompt_entities":   single_entities,
        "separate_prompt_entities": separate_entities,
        "effective_rate_raw":       eff_rate.get("effective_rate_raw"),
        "effective_rate_notes":     eff_rate.get("notes", ""),
        "benchmark_comparison":     benchmark_comparison,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A
# ═══════════════════════════════════════════════════════════════════════════════

def ask(question: str, doc_id: str, index_name: str) -> Dict[str, Any]:
    """Answer a question about an already-indexed document."""
    print(f"[Pipeline] Q&A: {question[:80]}")
    chunks = rag.retrieve(question, doc_id=doc_id, index_name=index_name)
    result = bedrock.ask_question(question, chunks)
    print(f"[Pipeline] Answer ready (chunks_used={result['chunks_used']})")
    return result
