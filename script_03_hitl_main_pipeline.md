# Script 03 — `hitl_main_pipeline.py`

---

## Opening

Now we're at the heart of the system — `hitl_main_pipeline.py`. This is the core business logic file for the entire HITL pipeline. It gets called from the two routes in `main.py` that we just covered, and it's where everything actually happens: building the review popup, persisting corrections, writing the audit trail, checking whether all reviews are done, and resuming the Step Function. Let's go through it section by section.

---

## Imports and AWS Client Setup

```python
from backend.src.storage.dynamodb_all import (
    get_extraction,
    get_evaluation,
    update_separate_prompt_entities,
    write_evaluation_record,
    update_hitl_feedback,
)

from backend.src.services import settings
import boto3
import json

sf = boto3.client("stepfunctions")

dynamodb = boto3.resource("dynamodb", region_name=settings.AWS_REGION)
table = dynamodb.Table(settings.DYNAMODB_TABLE)
```

We're importing five functions from `dynamodb_all.py`. We'll go deep on two of those — `update_separate_prompt_entities` and `update_hitl_feedback` — in the next script. The read functions `get_extraction` and `get_evaluation` were covered in the earlier DynamoDB session.

We also set up two AWS clients directly in this file. The `sf` Step Functions client is what we use later to resume the paused execution. The `dynamodb` resource and table handle direct writes that this file manages on its own — specifically the audit log records.

---

## `save_hitl_action` — The Audit Log

```python
def save_hitl_action(
    *,
    doc_id: str,
    field: str,
    original_value,
    corrected_value,
    reviewer_id: str | None,
    comments: str | None,
):
    table.put_item(
        Item={
            "PK": f"HITL#{doc_id}",
            "SK": f"ACTION#{datetime.utcnow().isoformat()}",
            "field": field,
            "original_value": original_value,
            "corrected_value": corrected_value,
            "reviewer_id": reviewer_id,
            "comments": comments,
            "source": "human_validated"
        }
    )
```

Every time a human corrects a field, we write a permanent audit record. The sort key is `ACTION#<timestamp>` — using a timestamp in the SK means every single correction gets its own record. If the same field is corrected twice, both records are kept. If two different reviewers touch the same document, both are tracked. You can always query `PK = HITL#<doc_id>` and get back the full chronological history of every human action on that document.

The `source: "human_validated"` flag is important downstream — it signals that this value came from a human, not the model. Anything reading these records later can immediately tell the difference.

Also notice the `*` in the function signature — all parameters are keyword-only. Callers must pass every argument by name, never positionally. That's a defensive pattern that prevents bugs when argument order changes over time, which matters a lot for an audit-critical function like this.

---

## `entity_needs_review`

```python
def entity_needs_review(judge_block: Dict[str, Any]) -> bool:
    label = (judge_block.get("label") or "").upper()

    if label in ("REVIEW", "INCORRECT"):
        return True

    return False
```

Simple utility — checks whether a judge block warrants HITL. Label-based check only. We deliberately kept this simple for the POC. If you look at the older version of this logic elsewhere in the codebase, it also checked confidence scores, evidence match ratios, ambiguity thresholds. We stripped all of that out here to keep things readable and easy to debug. Easy to evolve later.

---

## `get_hitl_task_token`

```python
def get_hitl_task_token(doc_id: str):
    resp = table.get_item(
        Key={
            "PK": f"HITL#{doc_id}",
            "SK": "TASK"
        }
    )
    return resp.get("Item", {}).get("task_token")
```

This reads back the task token that `hitl_wait_stub.py` stored. Same key structure — `HITL#<doc_id>` with `SK = TASK`. We call this in two places: once when building the popup payload, and once inside the resolution function when we're ready to resume the Step Function. If the record doesn't exist it returns `None` gracefully, and we check for that before calling `SendTaskSuccess`.

---

## `collect_candidates` — Building the Reviewer's Options

```python
def collect_candidates(
    field: str,
    extraction: Dict[str, Any],
    evaluation: Dict[str, Any]
) -> List[Any]:
    candidates: List[Any] = []

    # 1) Current extracted value
    original_val = extraction.get("_original_values", {}).get(field)
    if original_val not in (None, ""):
        candidates.append(original_val)

    # 2) Judge alternative candidates
    pem = extraction.get("per_entity_metrics", {})
    judge = pem.get(field, {}).get("judge", {})
    for c in judge.get("alt_candidates", []):
        if c not in candidates:
            candidates.append(c)

    # 3) Regex fallback candidates
    regex_candidates = extraction.get("_regex_candidates", {}).get(field, [])
    for r in regex_candidates:
        if r not in candidates:
            candidates.append(r)

    # Always allow manual entry
    candidates.append("manual_input")

    return candidates
```

This builds the list of options the reviewer sees in the popup. The goal is to give them the best possible set of choices so that in most cases they can just click the right one rather than having to type.

We pull from three sources in priority order. First, the original extracted value before the model flagged it — stored in `_original_values`. Second, any alternative candidates the LLM judge suggested — the judge model sometimes produces multiple possible values when it's uncertain and those are stored in `alt_candidates`. Third, regex-based fallback values from `_regex_candidates` — these are pattern-matched values that may be more structurally reliable even when the LLM wasn't confident.

We deduplicate throughout so the same value never shows up twice. And we always append `"manual_input"` at the end — that's a sentinel value the frontend recognizes to show a free text input field. The reviewer always has the option to type something none of the candidates captured.

---

## `build_hitl_popup_payload` — Serving the GET Route

```python
def build_hitl_popup_payload(doc_id: str, field: str) -> Dict[str, Any]:
    extraction = get_extraction(doc_id)
    evaluation = get_evaluation(doc_id)

    if not extraction or not evaluation:
        return {"ok": False, "error": "Missing extraction or evaluation."}

    pem = extraction.get("per_entity_metrics", {})
    judge = pem.get(field, {}).get("judge", {})
    task_token = get_hitl_task_token(doc_id)

    return {
        "ok": True,
        "doc_id": doc_id,
        "field": field,
        "task_token": task_token,
        "candidates": collect_candidates(field, extraction, evaluation),
        "evaluation_context": {
            "label": judge.get("label"),
            "final_score": judge.get("final_score"),
            "evidence": judge.get("evidence"),
            "rationale": judge.get("rationale"),
        },
    }
```

This is called directly from the GET route in `main.py` when the reviewer clicks the Review button in the frontend. It reads the extraction and evaluation records from DynamoDB and builds the popup response.

The `evaluation_context` block is particularly important for the reviewer experience. It surfaces the judge's label, confidence score, the evidence it found in the document, and its rationale for being uncertain. This gives the reviewer actual context about why a field was flagged — not just a list of options with no explanation. That context is what helps a reviewer make a fast, confident decision.

---

## `apply_hitl_resolution` — The Core of Everything

```python
def apply_hitl_resolution(
    *,
    doc_id: str,
    field: str,
    corrected_value: Any,
    reviewer_id: Optional[str] = None,
    comments: Optional[str] = None,
    task_token: Optional[str] = None,  # intentionally ignored
) -> Dict[str, Any]:
```

This is the most important function in the entire HITL pipeline. Called from the POST route in `main.py` when the reviewer submits a correction. Let's go through it step by step.

Notice `task_token` in the signature — the comment says intentionally ignored. The frontend can pass a token in the request body, but we never use it. We always retrieve the token fresh from DynamoDB server-side. We don't trust a token that came in from outside — the authoritative one is the one we stored ourselves when the Step Function started.

```python
    try:
        extraction = get_extraction(doc_id)
        evaluation = get_evaluation(doc_id)

        if not extraction:
            return {"ok": False, "error": "Extraction not found."}

        timestamp = datetime.utcnow().isoformat()
        original_value = extraction.get(field)

        update_separate_prompt_entities(
            doc_id=doc_id,
            updates={field: corrected_value}
        )
```

We read the current extraction record. We capture the original value before overwriting — we need it for the audit log. Then immediately we call `update_separate_prompt_entities` from `dynamodb_all.py` to write the corrected value back into the extraction record. We'll look at exactly how that function works in the next script.

```python
        sep = extraction.get("separate_prompt_entities", {})
        per_entity_metrics = sep.get("per_entity_metrics", {})

        if field in per_entity_metrics:
            per_entity_metrics[field]["judge"]["label"] = "HUMAN_ACCEPTED"
            per_entity_metrics[field]["source"] = "human_validated"

        save_hitl_action(
            doc_id=doc_id,
            field=field,
            original_value=original_value,
            corrected_value=corrected_value,
            reviewer_id=reviewer_id,
            comments=comments,
        )
```

We update the in-memory judge label to `HUMAN_ACCEPTED` and mark the source as `human_validated`. Then we write the audit record — that's `save_hitl_action` which we covered earlier — timestamped, before and after values, reviewer ID, comments. Permanent record.

```python
        extraction_after = get_extraction(doc_id)
        sep_after = extraction_after.get("separate_prompt_entities", {})
        pem_after = sep_after.get("per_entity_metrics", {})

        remaining_reviews: List[str] = []

        for f, meta in pem_after.items():
            value = sep_after.get(f)

            if value in (None, ""):
                remaining_reviews.append(f)
```

This is the completion check — and I want to be explicit about this. We re-read the extraction record fresh from DynamoDB after our write. Then we loop over every field in `per_entity_metrics` and check one thing: does this field have a non-empty value? That's the entire check. We are not re-evaluating judge labels. We are not checking confidence scores. We're just asking — does every field have a value?

This is a deliberate POC-level simplification. It's easy to understand, easy to debug, and easy to demonstrate. In a production version you'd want more nuance — checking specifically that all originally-flagged fields have been reviewed. But for where this product is right now, this works and it's clean.

```python
        task_token = get_hitl_task_token(doc_id)

        if task_token and not remaining_reviews:
            sf.send_task_success(
                taskToken=task_token,
                output=json.dumps({
                    "doc_id": doc_id,
                    "status": "all_reviews_completed",
                    "source": "human_validated",
                })
            )

            table.delete_item(
                Key={
                    "PK": f"HITL#{doc_id}",
                    "SK": "TASK"
                }
            )
```

If `remaining_reviews` is empty and we have a token — we resume the Step Function. `sf.send_task_success` is the call that wakes up the paused execution. We pass back a simple output with the `doc_id` and status. The Step Function receives this, moves to its End state, and the execution completes.

Immediately after, we delete the task token record from DynamoDB. Step Functions tokens are single-use — once you've called `SendTaskSuccess`, the token is spent. Leaving a stale one around would be misleading and confusing during debugging. Clean it up.

```python
        return {
            "ok": True,
            "doc_id": doc_id,
            "field": field,
            "value": corrected_value,
            "remaining_reviews": remaining_reviews,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "doc_id": doc_id,
            "error": f"HITL failure: {e}",
        }
```

The response includes `remaining_reviews` — the frontend uses this to know whether more fields still need attention or whether everything is resolved. The `traceback.print_exc()` in the exception handler sends the full stack trace to CloudWatch logs, which is important for debugging in Lambda and ECS environments where you don't have an interactive console.

---

## Closing

That's `hitl_main_pipeline.py`. It's the orchestrator of the entire review loop — builds the popup, persists corrections, writes the audit trail, checks completion, resumes the Step Function. Everything connects through this file.

Next we'll look at `dynamodb_all.py` — specifically the two storage functions this file calls into: `update_separate_prompt_entities` and `update_hitl_feedback`. Now that you've seen them being called, you'll have the full context for why they're built the way they are.

---
