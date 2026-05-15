# SCRIPT 2 — comprehend.py
### "PII Redaction via Amazon Comprehend"

---

## OPENING

"Now that we have the raw text out of the PDF, the very next thing we do before storing or processing any of it is redact PII — personally identifiable information. That's what `comprehend.py` handles. This is Step 2 of the data pipeline.

The reason we do this immediately after Textract and before anything else — before embeddings, before extraction — is compliance. Merchant statements can contain business owner names, addresses, tax IDs, and account numbers. We never want raw PII sitting in OpenSearch or being passed to an LLM. So we strip it here, at the earliest possible point in the pipeline.

Let me walk through how we do that."

---

## IMPORTS & CONFIG

```python
import boto3
from typing import List, Tuple
import time
import settings

_comprehend = boto3.client("comprehend", region_name=settings.AWS_REGION)
_s3 = boto3.client("s3", region_name=settings.AWS_REGION)

MASK_FORMAT = "[[REDACTED:{type}]]"
MIN_SCORE = 0.8
MAX_BYTES = 4000
```

"We initialize two clients — Comprehend for PII detection, and S3 for saving the redacted output. Three constants drive the redaction behavior:

`MASK_FORMAT` is the token we substitute for any detected PII. We chose a structured format — `[[REDACTED:NAME]]`, `[[REDACTED:SSN]]` — instead of just `***` or a blank, because it preserves context. Downstream components can still see where PII was and what type it was, without seeing the actual value.

`MIN_SCORE` is 0.8 — we only redact entities that Comprehend is at least 80% confident about. This prevents over-redacting things like common words that might look like names in certain contexts.

`MAX_BYTES` is 4000. This is critical — Amazon Comprehend has a hard limit of 5,000 bytes per API call. We set our chunk size to 4,000 to stay safely under that limit with a buffer for multi-byte characters."

---

## _chunk_text

```python
def _chunk_text(text: str) -> List[Tuple[str, int]]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + MAX_BYTES
        chunks.append((text[start:end], start))
        start = end

    return chunks
```

"Merchant statements can be several pages long — easily 10,000 to 50,000 characters. Since Comprehend has that byte limit, we need to split the text into chunks before sending it.

Each chunk is returned as a tuple of `(chunk_text, start_offset)`. The `start_offset` is the character position of that chunk within the original full text. We need this because Comprehend returns entity offsets relative to the chunk — not the full document. Without tracking the offset, we wouldn't know where to apply the redaction in the original text."

---

## _detect_pii

```python
def _detect_pii(text: str, metrics=None) -> List[dict]:
    try:
        api_start = time.time()
        resp = _comprehend.detect_pii_entities(Text=text, LanguageCode="en")
        api_latency = time.time() - api_start

        if metrics:
            metrics.log_api("comprehend", latency_sec=api_latency, success=True, ...)

        return resp.get("Entities", [])

    except Exception as e:
        print(f"[Comprehend] Detection error: {e}")

        if metrics:
            metrics.log_api("comprehend", latency_sec=0, success=False, ...)

        return []
```

"This private function makes a single Comprehend API call on one chunk of text. We call `detect_pii_entities` — Comprehend has other entity detection modes like `detect_entities` for general named entities, but we specifically want the PII-focused model which is trained to identify things like SSNs, credit card numbers, addresses, and phone numbers.

We wrap the call in a try/except and return an empty list on failure. That's a deliberate design choice — if Comprehend fails on one chunk, we don't crash the entire pipeline. We log the failure via the metrics system and continue. Some redaction is better than no processing at all.

We also track the latency around the API call and log it to our metrics manager if one is provided."

---

## _apply_redaction

```python
def _apply_redaction(text: str, spans: List[Tuple[int, int, str]]) -> str:
    spans = sorted(spans, key=lambda x: x[0])
    result = []
    last = 0

    for begin, end, pii_type in spans:
        result.append(text[last:begin])
        result.append(MASK_FORMAT.format(type=pii_type))
        last = end

    result.append(text[last:])
    return "".join(result)
```

"This is the core redaction logic. We receive a list of spans — each span is a `(begin_offset, end_offset, pii_type)` tuple — and we replace each one with the mask token.

We sort the spans by their start position first. This is important — if they overlap or are out of order, the string slicing would produce incorrect output. Then we walk through the text, copying everything before each span unchanged, inserting the mask token in place of the span, and advancing our position cursor.

We build the result as a list of strings and join at the end — this is significantly more efficient than string concatenation in a loop, which would create a new string object on every iteration."

---

## redact_pii — Main Entrypoint

```python
def redact_pii(text: str, metrics=None) -> str:
    chunks = _chunk_text(text)
    all_spans: List[Tuple[int, int, str]] = []

    for chunk_text, chunk_start in chunks:
        entities = _detect_pii(chunk_text, metrics=metrics)

        for e in entities:
            if float(e.get("Score", 0)) >= MIN_SCORE:
                all_spans.append(
                    (
                        e["BeginOffset"] + chunk_start,
                        e["EndOffset"] + chunk_start,
                        e["Type"],
                    )
                )

    redacted = _apply_redaction(text, all_spans)
    print(f"[Comprehend] Redacted {len(all_spans)} PII entity span(s)")
    return redacted
```

"This is the public function that `data_main_pipeline.py` calls. Here's where all the pieces come together.

We chunk the text, call Comprehend on each chunk, and for each entity returned we check the confidence score against our `MIN_SCORE` threshold of 0.8. If it passes, we take `BeginOffset` and `EndOffset` from Comprehend — but critically, we add `chunk_start` to both values to translate from chunk-relative positions back to full-document positions. That's the payoff from tracking the offset in `_chunk_text`.

We collect all qualifying spans across all chunks into `all_spans`, then make a single call to `_apply_redaction` on the original full text. The result is a clean, fully redacted string."

---

## save_redacted

```python
def save_redacted(redacted_text: str, doc_id: str, filename: str) -> str:
    key = f"{settings.S3_PREFIX_OUT}/{doc_id}_{filename.replace('.pdf', '')}_redacted.txt"

    _s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=redacted_text.encode("utf-8"),
    )

    print(f"[Comprehend] Saved redacted text → s3://{settings.S3_BUCKET}/{key}")
    return key
```

"Finally, we save the redacted text to S3 with a clear naming convention — `doc_id` + `filename` + `_redacted.txt`. Keeping the `doc_id` in the key makes it easy to look up the redacted file for any document later. We encode as UTF-8 before uploading and return the S3 key so `data_main_pipeline.py` can store it in DynamoDB.

So that's `comprehend.py` — chunking, detection, span correction, and redaction, all chained cleanly. The output of this file is what gets passed to everything downstream — embeddings, extraction, the LLM. Nothing downstream ever sees raw PII."

---
