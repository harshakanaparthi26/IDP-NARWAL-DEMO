# pipeline.py
"""
Main pipeline orchestrator.

All pipeline steps are called from here in order.
All print() statements that appear in the UI live here.

DynamoDB writes are non-blocking — pipeline never fails because of DynamoDB.

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
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key

import settings
import textract
import comprehend
import rag
import bedrock


# ═══════════════════════════════════════════════════════════════════════════════
# DynamoDB client (module-level, lazy)
# ═══════════════════════════════════════════════════════════════════════════════

_dynamo_resource = None

def _get_dynamo_table():
    global _dynamo_resource
    if _dynamo_resource is None:
        _dynamo_resource = boto3.resource("dynamodb", region_name=settings.AWS_REGION)
    return _dynamo_resource.Table(settings.DYNAMODB_TABLE)


def _safe_decimal(obj: Any) -> Any:
    """
    Recursively convert floats to Decimal for DynamoDB.
    DynamoDB does not accept Python float — must be Decimal.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _safe_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_decimal(i) for i in obj]
    return obj


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Non-blocking write ────────────────────────────────────────────────────────

def _dynamo_write(item: Dict) -> None:
    """
    Fire-and-forget DynamoDB put_item.
    Never raises — pipeline continues even if DynamoDB is unavailable.
    """
    try:
        table = _get_dynamo_table()
        table.put_item(Item=_safe_decimal(item))
    except Exception as e:
        print(f"[DynamoDB] Write failed (non-blocking): {e}")


def _dynamo_update(pk: str, sk: str, updates: Dict) -> None:
    """
    Update specific fields on an existing item.
    Never raises.
    """
    try:
        table  = _get_dynamo_table()
        expr   = "SET " + ", ".join(f"#f{i} = :v{i}" for i in range(len(updates)))
        names  = {f"#f{i}": k for i, k in enumerate(updates.keys())}
        values = {f":v{i}": _safe_decimal(v) for i, v in enumerate(updates.values())}
        table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )
    except Exception as e:
        print(f"[DynamoDB] Update failed (non-blocking): {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# DynamoDB record writers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_meta_start(
    doc_id:    str,
    doc_name:  str,
    s3_raw_key: str,
    merchant_group:  Optional[str],
    volume_tier_tsg: Optional[str],
) -> None:
    """Write initial metadata record when pipeline starts."""
    _dynamo_write({
        "PK":         f"DOC#{doc_id}",
        "SK":         "META",
        "doc_id":     doc_id,
        "doc_name":   doc_name,
        "s3_raw_key": s3_raw_key,
        "status":     "PROCESSING",
        "created_at": _now_iso(),
        "merchant_group":  merchant_group  or "",
        "volume_tier_tsg": volume_tier_tsg or "",
        # GSI1 — query all docs processed on a given date
        "GSI1PK": f"DATE#{_today()}",
        "GSI1SK": f"DOC#{doc_id}",
    })
    print(f"[DynamoDB] Meta record created: DOC#{doc_id}")


def _write_meta_complete(
    doc_id:       str,
    s3_text_key:  str,
    index_name:   str,
    char_count:   int,
    table_count:  int,
) -> None:
    """Update metadata record when pipeline completes successfully."""
    _dynamo_update(
        pk=f"DOC#{doc_id}",
        sk="META",
        updates={
            "status":      "COMPLETE",
            "completed_at": _now_iso(),
            "s3_text_key": s3_text_key,
            "opensearch_index_name": index_name,
            "char_count":  char_count,
            "table_count": table_count,
        },
    )
    print(f"[DynamoDB] Meta record updated: COMPLETE")


def _write_extraction_comparison(
    doc_id:             str,
    single_entities:    Dict,
    separate_entities:  Dict,
    effective_rate_raw: Optional[float],
    benchmark_comparison: Optional[Dict],
) -> None:
    """Write extraction comparison record after both methods complete."""

    # Build match flags
    entities = ["total_amount", "total_transactions_count", "total_fees"]
    match_flags = {
        f"match_{e}": single_entities.get(e) == separate_entities.get(e)
        for e in entities
    }

    # Pull judge verdicts per entity from separate prompts
    judge_verdicts = {
        e: ((separate_entities.get("per_entity_metrics") or {}).get(e) or {}).get("judge", {}).get("label")
        for e in entities
    }

    # Flatten single + separate for storage (drop per_entity_metrics — too nested)
    single_flat = {k: v for k, v in single_entities.items() if k not in ("method", "judge_result")}
    separate_flat = {k: v for k, v in separate_entities.items() if k not in ("method", "per_entity_metrics")}

    item = {
        "PK":        f"DOC#{doc_id}",
        "SK":        "EXTRACTION_COMPARISON",
        "doc_id":    doc_id,
        "created_at": _now_iso(),
        "single_prompt":   single_flat,
        "separate_prompts": separate_flat,
        "effective_rate_raw": effective_rate_raw,
        "match_flags":     match_flags,
        "judge_verdicts":  judge_verdicts,
    }

    if benchmark_comparison:
        item["benchmark_raw"]          = benchmark_comparison.get("benchmark_raw")
        item["delta_vs_benchmark_raw"] = benchmark_comparison.get("delta_vs_benchmark_raw")
        item["merchant_group"]         = benchmark_comparison.get("merchant_group", "")
        item["volume_tier_tsg"]        = benchmark_comparison.get("volume_tier_tsg", "")
        item["volume_tier_id"]         = benchmark_comparison.get("volume_tier_id")

    _dynamo_write(item)
    print(f"[DynamoDB] Extraction comparison record written: DOC#{doc_id}")


def _write_error(doc_id: str, step: str, error: Exception) -> None:
    """
    Write an error record when any pipeline step fails.
    GSI2 lets you query all failed documents across the system.
    """
    ts = _now_iso()
    _dynamo_write({
        "PK":           f"DOC#{doc_id}",
        "SK":           f"ERROR#{step}#{ts}",
        "doc_id":       doc_id,
        "step":         step,
        "error_message": str(error),
        "traceback":    traceback.format_exc()[:2000],   # cap at 2KB
        "created_at":   ts,
        # GSI2 — query all errors across all documents
        "GSI2PK": "STATUS#ERROR",
        "GSI2SK": f"DATE#{ts}",
    })
    # Also flip META status to ERROR
    _dynamo_update(
        pk=f"DOC#{doc_id}",
        sk="META",
        updates={"status": "ERROR", "failed_step": step, "error_message": str(error)[:500]},
    )
    print(f"[DynamoDB] Error record written: step={step}")


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
    return sorted(bench_df["MERCHANT_GROUP"].dropna().astype(str).str.strip().unique().tolist())


def get_volume_tiers(bench_df: pd.DataFrame, merchant_group: str) -> List[str]:
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
# Effective rate + benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def compute_effective_rate(entities: Dict[str, Any]) -> Dict[str, Any]:
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


def compare_with_benchmark(
    entities:        Dict[str, Any],
    merchant_group:  str,
    volume_tier_tsg: str,
    bench_df:        pd.DataFrame,
) -> Dict[str, Any]:
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

    if "PROCESS_MONTH" in df.columns:
        df = df[df["PROCESS_MONTH"] == df["PROCESS_MONTH"].max()]
    if "LOAD_DATE" in df.columns:
        df = df[df["LOAD_DATE"] == df["LOAD_DATE"].max()]

    pick      = df.iloc[0]
    bench_raw = float(pick["BENCHMARK"]) if pd.notna(pick.get("BENCHMARK")) else None
    delta     = (er_raw - bench_raw) if (er_raw is not None and bench_raw is not None) else None

    print(f"[Pipeline] Benchmark: '{mg}' / '{vt}' → benchmark={bench_raw}  delta={delta}")

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

    DynamoDB writes happen at:
      - Pipeline start      → META record (status=PROCESSING)
      - Each step failure   → ERROR record + META status=ERROR
      - After extraction    → EXTRACTION_COMPARISON record
      - Pipeline complete   → META record updated (status=COMPLETE)

    All DynamoDB writes are non-blocking.
    """
    def status(msg: str) -> None:
        print(msg)
        if on_status:
            on_status(msg)

    doc_id = make_doc_id(filename)

    # ── 1. Textract ───────────────────────────────────────────────────────────
    status(f"[1/4] Extracting text with Amazon Textract: {filename}")
    try:
        textract_out = textract.process_document(file, filename)
        if textract_out.get("status") != "success":
            raise RuntimeError(textract_out.get("error", "Textract failed"))
    except Exception as e:
        _write_error(doc_id, "textract", e)
        return {"error": str(e), "doc_id": doc_id, "doc_name": filename}

    raw_text = textract_out["text"]
    tables   = textract_out["tables"]
    print(f"[Pipeline] Textract done: {len(raw_text):,} chars, {len(tables)} table(s)")

    # Write initial META record now that we have the S3 raw key
    _write_meta_start(
        doc_id          = doc_id,
        doc_name        = filename,
        s3_raw_key      = textract_out.get("s3_raw_key", ""),
        merchant_group  = merchant_group,
        volume_tier_tsg = volume_tier_tsg,
    )

    # ── 2. PII redaction ──────────────────────────────────────────────────────
    status(f"[2/4] Redacting PII with Amazon Comprehend: {filename}")
    try:
        redacted_text = comprehend.redact_pii(raw_text)
        s3_text_key   = comprehend.save_redacted(redacted_text, doc_id, filename)
    except Exception as e:
        _write_error(doc_id, "comprehend", e)
        return {"error": str(e), "doc_id": doc_id, "doc_name": filename}

    # ── 3. RAG index ──────────────────────────────────────────────────────────
    status(f"[3/4] Building RAG index in OpenSearch: {filename}")
    try:
        index_name = rag.build_index(redacted_text, doc_id=doc_id, doc_name=filename)
    except Exception as e:
        _write_error(doc_id, "rag_index", e)
        return {"error": str(e), "doc_id": doc_id, "doc_name": filename}

    def retrieve_fn(query: str) -> List[Dict]:
        return rag.retrieve(query, doc_id=doc_id, index_name=index_name)

    # ── 4a. Single-prompt extraction ──────────────────────────────────────────
    status(f"[4/4] Extracting entities (single prompt): {filename}")
    try:
        general_chunks  = retrieve_fn("total amount total fees total transactions")
        single_entities = bedrock.extract_entities_single_prompt(general_chunks)
    except Exception as e:
        _write_error(doc_id, "extraction_single", e)
        single_entities = {"total_amount": None, "total_transactions_count": None, "total_fees": None}

    # ── 4b. Separate-prompt extraction ────────────────────────────────────────
    status(f"[4/4] Extracting entities (separate prompts): {filename}")
    try:
        separate_entities = bedrock.extract_entities_separate_prompts(retrieve_fn)
    except Exception as e:
        _write_error(doc_id, "extraction_separate", e)
        separate_entities = {"total_amount": None, "total_transactions_count": None, "total_fees": None}

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

    # ── DynamoDB: extraction comparison record ────────────────────────────────
    _write_extraction_comparison(
        doc_id              = doc_id,
        single_entities     = single_entities,
        separate_entities   = separate_entities,
        effective_rate_raw  = eff_rate.get("effective_rate_raw"),
        benchmark_comparison = benchmark_comparison,
    )

    # ── DynamoDB: update META to COMPLETE ─────────────────────────────────────
    _write_meta_complete(
        doc_id      = doc_id,
        s3_text_key = s3_text_key,
        index_name  = index_name,
        char_count  = len(redacted_text),
        table_count = len(tables),
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
