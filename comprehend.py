# comprehend.py
"""
Step 2 — PII redaction via Amazon Comprehend.

Functions:
    redact_pii(text)  → redacted_text
    save_redacted(redacted_text, doc_id, filename)  → s3_key
"""
import boto3
from typing import List, Tuple

import settings


# ── AWS client ────────────────────────────────────────────────────────────────
_comprehend = boto3.client("comprehend", region_name=settings.AWS_REGION)
_s3         = boto3.client("s3",         region_name=settings.AWS_REGION)

MASK_FORMAT = "[[REDACTED:{type}]]"
MIN_SCORE   = 0.8
MAX_BYTES   = 4000   # Comprehend byte limit per call


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[Tuple[str, int]]:
    """Split text into (chunk, start_offset) pairs within Comprehend's byte limit."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_BYTES
        chunks.append((text[start:end], start))
        start = end
    return chunks


# ── PII detection ─────────────────────────────────────────────────────────────

def _detect_pii(text: str) -> List[dict]:
    """Call Comprehend PII detection. Returns raw entity list."""
    try:
        resp = _comprehend.detect_pii_entities(Text=text, LanguageCode="en")
        return resp.get("Entities", [])
    except Exception as e:
        print(f"[Comprehend] Detection error: {e}")
        return []


# ── Redaction ─────────────────────────────────────────────────────────────────

def _apply_redaction(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """Replace PII spans in text with masked tokens."""
    spans = sorted(spans, key=lambda x: x[0])
    result = []
    last = 0
    for begin, end, pii_type in spans:
        result.append(text[last:begin])
        result.append(MASK_FORMAT.format(type=pii_type))
        last = end
    result.append(text[last:])
    return "".join(result)


# ── Main entrypoint ───────────────────────────────────────────────────────────

def redact_pii(text: str) -> str:
    """
    Detect and redact PII from text using Amazon Comprehend.
    Handles large inputs by chunking automatically.
    Returns fully redacted text string.
    """
    chunks = _chunk_text(text)
    all_spans: List[Tuple[int, int, str]] = []

    for chunk_text, chunk_start in chunks:
        entities = _detect_pii(chunk_text)
        for e in entities:
            if float(e.get("Score", 0)) >= MIN_SCORE:
                all_spans.append((
                    e["BeginOffset"] + chunk_start,
                    e["EndOffset"]   + chunk_start,
                    e["Type"],
                ))

    redacted = _apply_redaction(text, all_spans)
    print(f"[Comprehend] Redacted {len(all_spans)} PII entity span(s)")
    return redacted


def save_redacted(redacted_text: str, doc_id: str, filename: str) -> str:
    """
    Save redacted text to S3.
    Returns the S3 key.
    """
    key = f"{settings.S3_PREFIX_OUT}{doc_id}_{filename.replace('.pdf', '')}_redacted.txt"
    _s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=redacted_text.encode("utf-8"),
    )
    print(f"[Comprehend] Saved redacted text → s3://{settings.S3_BUCKET}/{key}")
    return key
