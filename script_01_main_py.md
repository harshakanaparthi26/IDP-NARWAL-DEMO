# Script 01 — `main.py` (HITL Trigger + Routes)

---

## Opening

Alright, let's start with `main.py`. This file is the FastAPI entry point for the entire backend and was already covered in an earlier session for the document ingestion flow — Textract, Comprehend, evaluation, all of that. I'm not going to repeat any of that.

What I'm covering here are two things. First, the exact moment inside the ingest endpoint where the HITL pipeline gets triggered — this is the handoff point from everything we've already covered into the new territory we're walking through today. Second, the two API routes that the frontend calls during the review process. These are the entry points into everything else we're about to see.

---

## The HITL Trigger — Inside `/document/ingest`

```python
extraction = get_extraction(doc_id)

sep = extraction.get("separate_prompt_entities", {})
per_entity_metrics = sep.get("per_entity_metrics", {})

review_fields = []

for field, meta in per_entity_metrics.items():
    judge = meta.get("judge", {})
    label = (judge.get("label") or "").upper()
    value = sep.get(field)

    if label in {"REVIEW", "INCORRECT"} or value in (None, ""):
        review_fields.append(field)

print("DEBUG review_fields (EXACT UI SOURCE):", review_fields)
```

This block runs right after the evaluation pipeline has finished and written its results to DynamoDB. We read the extraction record back, loop over every field in `per_entity_metrics`, and collect anything where the judge came back with `REVIEW` or `INCORRECT`, or where the value itself is null or empty. That list is `review_fields` — these are the fields that need a human to look at them.

```python
import uuid

if review_fields:
    resp = sf.start_execution(
        stateMachineArn=settings.HITL_STATE_MACHINE_ARN,
        input=json.dumps({
            "doc_id": doc_id,
            "review_fields": review_fields,
        }),
        name=f"hitl-{doc_id}-{uuid.uuid4()}"
    )

    print("HITL STEP FUNCTION STARTED")
    print("Execution ARN:", resp["executionArn"])
```

If `review_fields` is non-empty, we call `sf.start_execution` — that's the boto3 Step Functions client — and we pass in the state machine ARN from our settings module and the `doc_id` plus `review_fields` as the execution input. The infrastructure behind this — the state machine itself, the IAM roles — is all provisioned by Terraform and will be covered in a separate Terraform KT session.

One specific thing worth calling out: the `name` parameter. Step Functions requires execution names to be unique within a given state machine. We use `hitl-{doc_id}-{uuid4()}` to guarantee that. Without the UUID suffix, if the same document ever gets reprocessed, the second execution would collide on the name and throw an error. The UUID makes sure that never happens.

The `executionArn` gets printed to logs — that gives you a direct reference to find this specific execution in the AWS console when you're debugging.

---

## The HITL GET Route

```python
@app.get("/hitl/{doc_id}/field/{field}")
def hitl_get_field_popup(doc_id: str, field: str):
    """
    Returns HITL popup payload.
    Step Function must ALREADY be running.
    """
    popup = build_hitl_popup_payload(
        doc_id=doc_id,
        field=field
    )

    return popup
```

This is the first of the two HITL routes. The frontend calls this when a reviewer clicks the Review button for a specific field. The `doc_id` and `field` come in as path parameters — FastAPI parses those automatically — and the handler does exactly one thing: it calls into `hitl_main_pipeline.py` to build the popup payload and returns it. We'll see exactly what that function does when we get to that file.

The docstring is worth noting — it says the Step Function must already be running. This route doesn't start anything, it just reads. If you hit this route before the Step Function has started, the task token won't exist in DynamoDB yet and the popup will come back without one. The ordering matters.

---

## The HITL POST Route

```python
@app.post("/hitl/{doc_id}/field/{field}")
def hitl_apply_field(
    doc_id: str,
    field: str,
    payload: Dict[str, Any]
):
    return apply_hitl_resolution(
        doc_id=doc_id,
        field=field,
        corrected_value=payload.get("corrected_value"),
        reviewer_id=payload.get("reviewer_id"),
        comments=payload.get("comments"),
        task_token=payload.get("task_token"),
    )
```

This is the second route — the one the frontend calls when the reviewer clicks Save after choosing or typing a corrected value. The request body comes in as `payload`, FastAPI deserializes it automatically, and we pull out the corrected value, the reviewer ID, any comments, and a task token if the frontend passed one.

All of that goes straight into `apply_hitl_resolution` in `hitl_main_pipeline.py`. These routes are intentionally thin — they do nothing except delegate to the pipeline logic. That keeps the API layer clean and keeps the real logic testable in isolation. We'll walk through exactly what `apply_hitl_resolution` does in detail when we get to that file.

---

## Closing

Those are the two entry points into the HITL system from the outside world. `main.py` starts the Step Function after evaluation, and these two routes are how the frontend communicates corrections back into the pipeline.

Next we'll look at what happens immediately after that Step Function starts — the wait Lambda in `hitl_wait_stub.py`.

---
