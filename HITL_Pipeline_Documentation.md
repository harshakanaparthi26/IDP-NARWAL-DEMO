# Phoenix Statement Reporting — HITL Pipeline Technical Documentation

**Project:** Phoenix Statement Reporting (POC)
**Client:** Worldpay
**Scope:** HITL Pipeline — Phase 1 Implementation
**Document Type:** Internal Technical Documentation
**Status:** POC — Active Development
**Audience:** Developers, Technical Stakeholders

---

> **Document Purpose**
>
> This document provides a complete technical reference for the Human-in-the-Loop (HITL) pipeline implemented in the Phoenix Statement Reporting project. It covers:
> - Current architecture and AWS resource inventory
> - End-to-end data flow with code-level explanations
> - All files and functions involved, and how they interact
> - DynamoDB schema and key design patterns
> - Known limitations and bugs in the current Phase 1 implementation
> - Recommended fixes and a proposed full production architecture

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is HITL and Why It Is Needed](#2-what-is-hitl-and-why-it-is-needed)
3. [AWS Resources Involved](#3-aws-resources-involved)
4. [Files Involved and Their Roles](#4-files-involved-and-their-roles)
5. [End-to-End HITL Flow](#5-end-to-end-hitl-flow)
6. [DynamoDB Schema and Key Design](#6-dynamodb-schema-and-key-design)
7. [Key Functions Reference](#7-key-functions-reference)
8. [Known Limitation — Second Review Button Does Not Resume Step Function](#8-known-limitation--second-review-button-does-not-resume-step-function)
9. [Future Architecture — Full Production HITL](#9-future-architecture--full-production-hitl)
10. [Summary](#10-summary)

---

## 1. Project Overview

Phoenix Statement Reporting is a Proof-of-Concept (POC) application built for Worldpay. Its primary purpose is to allow an Independent Software Vendor (ISV) to upload a merchant payment statement and receive an optimised onboarding quote.

### 1.1 What the Application Does

When a merchant statement PDF is uploaded, the pipeline performs the following steps automatically:

1. Extracts key financial entities from the statement using AWS Textract and LLM-based retrieval (RAG).
2. Evaluates extraction confidence using an LLM Judge that scores each extracted field.
3. Compares extracted interchange fees against TSG benchmarks and Visa/Mastercard public rates to identify anomalies.
4. Triggers a Human-in-the-Loop (HITL) review for any field the judge marks as `REVIEW` or `INCORRECT`.
5. Generates a best-price onboarding quote for the ISV once all reviews are resolved.

### 1.2 The Three Key Entities

The current HITL implementation covers three critical financial fields extracted from each merchant statement:

| Entity Field | Description | Why It Matters |
|---|---|---|
| `total_amount` | Total processed volume (dollar amount) | Baseline for quote calculation |
| `total_fees` | Total fees charged to the merchant | Key profitability metric |
| `total_transactions_count` | Number of transactions in the period | Volume tier classification |

> **Future Scope:** In a future phase, HITL coverage will be extended to Visa and Mastercard interchange program fee fields (program codes, percent rates, flat rates, etc.). The current architecture has been designed with this extension in mind.

---

## 2. What is HITL and Why It Is Needed

Human-in-the-Loop (HITL) is a design pattern where a human reviewer is inserted into an automated pipeline at decision points where the system's confidence is too low to proceed automatically.

In this project, the LLM Judge scores each extracted field on multiple quality dimensions. When the judge determines a value is uncertain, incorrect, or missing, the pipeline pauses and waits for a human to confirm or correct the value before continuing.

### 2.1 When Does HITL Trigger?

HITL is triggered by the LLM Judge evaluation result. A field is flagged for review if any of the following conditions are true:

| Condition | Meaning |
|---|---|
| Judge label = `REVIEW` | Model is uncertain about the extracted value |
| Judge label = `INCORRECT` | Model believes the extracted value is wrong |
| Judge label = `HALLUCINATION` | Model output has no supporting evidence in the document |
| Field value = `None` or empty string | Extraction returned no value at all |
| `evidence_match` score < 0.50 | Weak evidence linkage between source text and extracted value |
| `hallucination_likelihood` > 0.50 | High probability the value was fabricated |

### 2.2 What Happens During HITL

When a field is flagged:

1. The pipeline starts an AWS Step Functions execution that pauses and waits for a human callback (`waitForTaskToken` pattern).
2. The UI displays a blue **Review** button next to the flagged field.
3. The reviewer clicks the button, sees candidate values (extracted, judge alternatives, regex fallback, manual input), and selects or enters the correct value.
4. The corrected value is written back to DynamoDB and the Step Function receives a success callback, allowing the pipeline to continue.

---

## 3. AWS Resources Involved

The following AWS resources are provisioned via Terraform and participate in the HITL pipeline.

### 3.1 Resource Inventory

| Resource Type | Name / Identifier | Role in HITL |
|---|---|---|
| AWS Step Functions | `wp-phoenix-statement-reporting-state-machine` | Orchestrates the HITL pause-and-wait pattern using `waitForTaskToken` |
| Lambda — `hitl_wait` | `wp-phoenix-...-hitl-wait` | Receives the task token from Step Functions and stores it in DynamoDB |
| Lambda — `hitl_resolve` | `wp-phoenix-...-hitl-resolve` | Early placeholder Lambda; resolution is now handled by the FastAPI POST route (**not active**) |
| Lambda — `processor` | `wp-phoenix-...-processor-lambda` | General-purpose document processor Lambda |
| DynamoDB Table | `wp-phoenix-statement-reporting-table` | Stores task tokens, extraction records, evaluation records, and HITL audit logs |
| SageMaker Notebook | `wp-phoenix-...-notebook` (×4) | Development environment; hosts the FastAPI backend and Streamlit UI |
| Bedrock Guardrail | `merchant-statement-pii-guardrail` | Anonymises PII (names, phones, emails, addresses) in LLM inputs/outputs |
| S3 — Lambda Bucket | `wp-phoenix-statement-reporting-lambda-bucket` | Stores Lambda deployment packages (`.zip` files) |
| IAM Roles | `lambda-execution-role`, `step-functions-role`, `sagemaker-execution-role` | Least-privilege roles governing resource access |

### 3.2 Step Function State Machine Definition

The state machine has a single state: `WaitForHuman`. It uses the `waitForTaskToken` integration pattern — meaning it pauses execution indefinitely until the pipeline sends a `SendTaskSuccess` or `SendTaskFailure` callback with the stored token.

```json
{
  "Comment": "Phoenix HITL Wait-Only State Machine",
  "StartAt": "WaitForHuman",
  "States": {
    "WaitForHuman": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke.waitForTaskToken",
      "Parameters": {
        "FunctionName": "<hitl_wait_lambda_arn>",
        "Payload": {
          "task_token.$": "$$.Task.Token",
          "doc_id.$":     "$.doc_id"
        }
      },
      "End": true
    }
  }
}
```

> **Key Design Decision — `waitForTaskToken`:** This pattern means Step Functions does NOT poll. It suspends the execution and forwards the unique token to the Lambda. The pipeline must call `states:SendTaskSuccess` with that exact token to resume. This is ideal for asynchronous human review where the reviewer may take minutes, hours, or days to respond.

---

## 4. Files Involved and Their Roles

| File | Layer | Role |
|---|---|---|
| `main.py` | FastAPI Backend (Entry Point) | Exposes the REST API. Handles `/document/ingest` (starts Step Function if review fields detected), `GET /hitl/{doc_id}/field/{field}` (popup payload), and `POST /hitl/{doc_id}/field/{field}` (applies resolution). Imports from `hitl_main_pipeline.py`. |
| `hitl_main_pipeline.py` | Business Logic | Core HITL orchestration. Contains `build_hitl_popup_payload()` and `apply_hitl_resolution()`. Responsible for reading DynamoDB, computing remaining reviews, and calling `SendTaskSuccess` when all reviews are done. |
| `hitl_wait_stub.py` | Lambda Function | Deployed as the `hitl-wait` Lambda. Receives the `task_token` from Step Functions and writes it to DynamoDB (`PK=HITL#doc_id`, `SK=TASK`) for later retrieval by the resolution logic. |
| `hitl_resolve_handler.py` | **Lambda (Placeholder — Not Active)** | Early prototype Lambda that called `SendTaskSuccess` directly. The active resolution path is the FastAPI POST route in `main.py` calling `apply_hitl_resolution()`. This file is no longer part of the active flow. |
| `dynamodb_all.py` | Data Access Layer | Unified DynamoDB access module. Provides `get_extraction()`, `get_evaluation()`, `update_separate_prompt_entities()`, `write_evaluation_record()`, and `update_hitl_feedback()`. All reads and writes go through this module. |
| `3_Extracted_with_llm_judge.py` | Streamlit Frontend (UI Page) | Renders the extracted entities table with judge verdicts. Displays blue **Review** buttons for flagged fields. Opens the HITL modal popup when a reviewer clicks Review. Calls the FastAPI backend to fetch candidates and submit corrections. |
| `terrafrom_main.txt` | Infrastructure (Terraform) | Declares all AWS resources: Step Functions state machine, Lambda functions, DynamoDB table reference, IAM roles and policies, Bedrock guardrail, and SageMaker notebooks. |

> **Note on inactive files:**
> - `hitl_resolve_handler.py` — early prototype, superseded by the FastAPI POST route. Can be archived.
> - `hitl_main.py` — parallel/earlier version of the HITL logic, superseded by `hitl_main_pipeline.py`. Not imported by `main.py`. Can be archived.

---

## 5. End-to-End HITL Flow

### 5.1 High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND (UI)                     │
│    Page: 3_Extracted_with_llm_judge.py                         │
└──────────────────┬──────────────┬──────────────────────────────┘
   GET /hitl/{id}/field  POST /hitl/{id}/field    POST /document/ingest
          │                     │                         │
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND  (main.py)                    │
│  hitl_get_field_popup()  hitl_apply_field()  document_ingest() │
│               │                  │                 │            │
│               └──────────────────┘     (Step Functions         │
│                        ┃                start_execution)        │
└────────────────────────╋────────────────────────┬──────────────┘
                         ┃                         │
┌────────────────────────┸──────────┐  ┌──────────────────────────┐
│  hitl_main_pipeline.py            │  │  AWS Step Functions      │
│  build_hitl_popup_payload()       │  │  State: WaitForHuman     │
│  apply_hitl_resolution()          │  │  (waitForTaskToken)      │
└────────┬──────────────────────────┘  └───────────┬──────────────┘
         │                                          │ invokes Lambda
┌────────┸──────────────────┐          ┌────────────────────────────┐
│  dynamodb_all.py           │          │  hitl_wait_stub.py         │
│  (Data Access Layer)       ┠──────────│  stores task_token in DDB  │
│  get_extraction()          │          └────────────────────────────┘
│  update_separate_prompt..  │
└────────┬───────────────────┘
         │
┌────────┸───────────────────────────────────────────┐
│    DYNAMODB TABLE                                   │
│    wp-phoenix-statement-reporting-table             │
└─────────────────────────────────────────────────────┘
```

### 5.2 Step-by-Step Flow

---

#### Step 1 — Document Upload and Extraction (`main.py: document_ingest`)

When the ISV uploads a merchant statement PDF, the file goes through:
- Data pipeline (Textract)
- Embedding pipeline
- Retrieval pipeline (RAG + LLM extraction)
- Evaluation pipeline (LLM Judge scoring)

The evaluation result for each field is stored in DynamoDB with a `per_entity_metrics` block containing the judge label. The backend then scans for fields to flag:

```python
# main.py — inside document_ingest()
for field, meta in per_entity_metrics.items():
    judge = meta.get('judge', {})
    label = (judge.get('label') or '').upper()
    value = sep.get(field)
    if label in {'REVIEW', 'INCORRECT'} or value in (None, ''):
        review_fields.append(field)
```

---

#### Step 2 — Step Function Start (`main.py: document_ingest`)

If `review_fields` is non-empty, the backend starts a Step Functions execution:

```python
# main.py
if review_fields:
    resp = sf.start_execution(
        stateMachineArn=settings.HITL_STATE_MACHINE_ARN,
        input=json.dumps({'doc_id': doc_id, 'review_fields': review_fields}),
        name=f'hitl-{doc_id}-{uuid.uuid4()}'   # always unique
    )
```

---

#### Step 3 — Token Storage (`hitl_wait_stub.py: handler`)

The Step Function invokes the `hitl-wait` Lambda and forwards the `task_token` in the payload. The Lambda stores this token in DynamoDB for later retrieval:

```python
# hitl_wait_stub.py
def handler(event, context):
    doc_id     = event['doc_id']
    task_token = event['task_token']
    table.put_item(Item={
        'PK': f'HITL#{doc_id}',
        'SK': 'TASK',
        'task_token': task_token,
        'created_at': datetime.utcnow().isoformat(),
    })
```

---

#### Step 4 — Reviewer Opens the UI (`3_Extracted_with_llm_judge.py`)

The Streamlit frontend renders the extracted entities. Fields labelled `REVIEW` or `INCORRECT` show a blue **Review** button instead of the raw value:

```python
# 3_Extracted_with_llm_judge.py
def needs_review(judge):
    return judge.get('label') in ('REVIEW', 'INCORRECT')

# In the render loop:
if needs_review(judge):
    if st.button('🔵 Review', key=f'review_{field}'):
        popup = fetch_hitl_popup(field)   # GET /hitl/{doc_id}/field/{field}
        ss['hitl_modal'] = {'open': True, 'field': field, 'payload': popup}
        st.rerun()
```

---

#### Step 5 — HITL Popup Payload (`hitl_main_pipeline.py: build_hitl_popup_payload`)

When the reviewer clicks Review, the frontend calls `GET /hitl/{doc_id}/field/{field}`. The backend reads DynamoDB and returns candidate values:

```python
# hitl_main_pipeline.py
def build_hitl_popup_payload(doc_id, field):
    extraction = get_extraction(doc_id)     # DynamoDB read
    evaluation = get_evaluation(doc_id)     # DynamoDB read
    candidates = collect_candidates(field, extraction, evaluation)
    task_token = get_hitl_task_token(doc_id)  # reads HITL#doc_id / TASK
    return {
        'ok': True,
        'candidates': candidates,           # options shown to reviewer
        'evaluation_context': {...},        # evidence, rationale, score
        'task_token': task_token,
    }
```

Candidate values are assembled in this priority order:

1. The original extracted value (before it was flagged as Review)
2. Alternative candidates suggested by the LLM Judge (`alt_candidates`)
3. Regex fallback candidates extracted from the raw document text
4. `manual_input` — reviewer can type a value directly

---

#### Step 6 — Reviewer Submits Correction (`hitl_main_pipeline.py: apply_hitl_resolution`)

When the reviewer clicks Save, the frontend calls `POST /hitl/{doc_id}/field/{field}`. The backend:

1. Writes the corrected value to `separate_prompt_entities` in DynamoDB
2. Updates the judge label to `HUMAN_ACCEPTED` for the resolved field
3. Appends a HITL audit record (`PK=HITL#doc_id`, `SK=ACTION#timestamp`)
4. Checks `remaining_reviews` — any field still `None` or empty
5. If `remaining_reviews` is empty, calls `sf.send_task_success()` with the stored token, then deletes the `HITL#doc_id / TASK` record

```python
# hitl_main_pipeline.py — apply_hitl_resolution()
remaining_reviews = [
    f for f, meta in pem_after.items()
    if sep_after.get(f) in (None, '')
]

task_token = get_hitl_task_token(doc_id)

if task_token and not remaining_reviews:
    sf.send_task_success(
        taskToken=task_token,
        output=json.dumps({'doc_id': doc_id, 'status': 'all_reviews_completed'})
    )
    table.delete_item(Key={'PK': f'HITL#{doc_id}', 'SK': 'TASK'})
```

---

## 6. DynamoDB Schema and Key Design

All data is stored in a single DynamoDB table using a single-table design. Table name: `wp-phoenix-statement-reporting-table`.

### 6.1 Record Types

| PK | SK | Purpose |
|---|---|---|
| `DOC#<doc_id>` | `EXTRACTION_COMPARISON` | Stores all extracted entities (`separate_prompt_entities`) including `per_entity_metrics` with judge scores. Updated by `apply_hitl_resolution` when a reviewer corrects a value. |
| `EVAL#<doc_id>` | `EVAL#LATEST` | Latest evaluation record for the document including entity scores, accuracy, and HITL feedback log. |
| `EVAL#<doc_id>` | `EVAL#<timestamp>` | Versioned historical evaluation snapshot written each time the evaluation pipeline runs. |
| `HITL#<doc_id>` | `TASK` | Holds the Step Function `task_token`. Written by the `hitl_wait_stub` Lambda when Step Function starts. **Deleted** by `apply_hitl_resolution` when all reviews are complete. |
| `HITL#<doc_id>` | `ACTION#<timestamp>` | Immutable HITL audit record. One record per reviewer correction. Contains `original_value`, `corrected_value`, `reviewer_id`, `comments`, `source=human_validated`. |
| `DOC#<doc_id>` | `META` | Document metadata: file name, S3 key, processing status, timestamps, and classification metadata (industry, volume tier, region). |

### 6.2 Key DynamoDB Access Patterns in HITL

| Operation | Function | Access Pattern |
|---|---|---|
| Read extraction for popup | `get_extraction(doc_id)` | `GetItem`: PK=`DOC#<id>`, SK=`EXTRACTION_COMPARISON` |
| Read task token | `get_hitl_task_token(doc_id)` | `GetItem`: PK=`HITL#<id>`, SK=`TASK` |
| Write corrected value | `update_separate_prompt_entities()` | `UpdateItem` with `SET` on `separate_prompt_entities.<field>` |
| Write audit record | `save_hitl_action()` | `PutItem`: PK=`HITL#<id>`, SK=`ACTION#<iso-timestamp>` |
| Delete token after resolution | `table.delete_item()` | `DeleteItem`: PK=`HITL#<id>`, SK=`TASK` |
| Read full evaluation history | `get_evaluation(doc_id)` | `Query`: PK=`EVAL#<id>`, returns all SK versions |

---

## 7. Key Functions Reference

| Function | File | Description |
|---|---|---|
| `document_ingest()` | `main.py` | Full ingestion entry point. Runs all pipeline stages, identifies review fields, and starts the Step Function execution. |
| `hitl_get_field_popup()` | `main.py` | FastAPI `GET` handler. Delegates to `build_hitl_popup_payload()` and returns the response to the frontend. |
| `hitl_apply_field()` | `main.py` | FastAPI `POST` handler. Delegates to `apply_hitl_resolution()` and returns the result. |
| `build_hitl_popup_payload()` | `hitl_main_pipeline.py` | Reads extraction and evaluation from DynamoDB. Builds the candidate list and evaluation context payload shown in the reviewer popup. |
| `collect_candidates()` | `hitl_main_pipeline.py` | Assembles the ordered list of candidate values from: original extracted value, judge `alt_candidates`, regex fallback values, and `manual_input` placeholder. |
| `apply_hitl_resolution()` | `hitl_main_pipeline.py` | Core resolution function. Saves the corrected value, updates PEM metadata, writes the audit record, checks remaining reviews, and conditionally calls `SendTaskSuccess`. |
| `get_hitl_task_token()` | `hitl_main_pipeline.py` | Reads the `task_token` from DynamoDB (`HITL#doc_id / TASK`). Returns `None` if the token has already been consumed or was never stored. |
| `save_hitl_action()` | `hitl_main_pipeline.py` | Writes an immutable audit record to DynamoDB for every reviewer correction (`PK=HITL#doc_id`, `SK=ACTION#timestamp`). |
| `handler()` | `hitl_wait_stub.py` | Lambda entry point. Extracts `doc_id` and `task_token` from the Step Function payload and persists the token to DynamoDB. |
| `get_extraction()` | `dynamodb_all.py` | `GetItem` for the `EXTRACTION_COMPARISON` record. Returns the full extraction dict including `separate_prompt_entities` and `per_entity_metrics`. |
| `update_separate_prompt_entities()` | `dynamodb_all.py` | Safe partial update — uses `SET` expressions to merge individual field updates into `separate_prompt_entities` without overwriting other fields. |
| `update_hitl_feedback()` | `dynamodb_all.py` | Appends HITL correction details into the `hitl` map on the `EVAL#LATEST` record. |

---

## 8. Known Limitation — Second Review Button Does Not Resume Step Function

> ⚠️ **Bug Summary**
>
> When multiple fields are flagged for review (e.g., `total_amount` and `total_fees`), the HITL flow works correctly for the **first** review button clicked. However, when the reviewer clicks the **second** review button, the corrected value is saved to DynamoDB but the Step Function is **not** resumed.
>
> **Root cause:** the `task_token` is deleted from DynamoDB after the first resolution, so it is unavailable (`None`) when the second field is resolved.

### 8.1 Observed Terminal Output

Captured during a test with `doc_id = 4_pii.pdf`, where both `total_amount` and `total_fees` were flagged:

```
DEBUG review_fields (EXACT UI SOURCE): ['total_amount', 'total_fees']
🚀 HITL STEP FUNCTION STARTED
🆔 Execution ARN: arn:aws:states:us-east-2:...:execution:...hitl-4_pii.pdf...

# --- Reviewer clicks Review for total_amount ---
DEBUG remaining_reviews: []          # <-- 0 remaining (fires too early)
DEBUG task_token: AQCcAAAAKg...      # <-- token present
****...$$$$...^^^^   (SendTaskSuccess fires ✅)

# --- Reviewer clicks Review for total_fees ---
DEBUG remaining_reviews: []
DEBUG task_token: None               # <-- token already deleted!
# SendTaskSuccess is NOT called. Only DB update happens. ❌
```

### 8.2 Root Cause Analysis

The bug has two contributing factors.

#### Factor 1 — Premature `remaining_reviews` Calculation

The `remaining_reviews` logic checks for fields where the value is `None` or empty string. However, after the first field is resolved, the second field may already have a value (even if still labelled `REVIEW`). This causes `remaining_reviews` to become empty after the **first** review — triggering `SendTaskSuccess` and deleting the token prematurely.

```python
# CURRENT (buggy) logic:
remaining_reviews = [
    f for f, meta in pem_after.items()
    if sep_after.get(f) in (None, '')   # only checks for empty values
]                                        # does NOT check judge label

# If total_fees already has a value (even if labelled REVIEW),
# it will NOT appear in remaining_reviews — token fires early.
```

#### Factor 2 — Token Deleted After First `SendTaskSuccess`

Once `SendTaskSuccess` fires, the `HITL#doc_id / TASK` record is deleted from DynamoDB. When the second reviewer submits, `get_hitl_task_token()` returns `None`, so no callback is sent to Step Functions.

### 8.3 Current Behaviour Summary

| Action | First Review Button | Second Review Button |
|---|---|---|
| Value saved to DynamoDB | ✅ Yes | ✅ Yes |
| HITL audit record written | ✅ Yes | ✅ Yes |
| `task_token` retrieved | ✅ Yes (token present) | ❌ No (token is `None`) |
| `SendTaskSuccess` called | ✅ Yes | ❌ No |
| Step Function resumed | ✅ Yes | ❌ No |

### 8.4 Recommended Fix

Two changes are needed.

#### Fix 1 — Correct the `remaining_reviews` Logic

Change the check from "field has no value" to "field still has a `REVIEW` or `INCORRECT` judge label AND has not been human-validated". This ensures the Step Function does not fire until all flagged fields have been explicitly approved by a human.

```python
# FIXED logic — check judge label and source, not just value emptiness
remaining_reviews = []
for f, meta in pem_after.items():
    judge_label = (meta.get('judge', {}).get('label') or '').upper()
    source      = meta.get('source', '')
    value       = sep_after.get(f)

    needs_review = (
        judge_label in {'REVIEW', 'INCORRECT', 'HALLUCINATION'}
        and source != 'human_validated'
    ) or value in (None, '')

    if needs_review:
        remaining_reviews.append(f)
```

#### Fix 2 — Do Not Delete the Token Until All Fields Are Resolved

The token deletion should only happen once `remaining_reviews` is truly empty with the corrected logic above. With Fix 1 in place, this will naturally be true only after all flagged fields have been human-validated.

> **Note on Step Function Design:** In the current Phase 1 design, one Step Function execution covers all flagged fields for a document. The execution must remain open until every field is resolved. The token must therefore persist in DynamoDB throughout the entire review session. It should be deleted **only** when the final reviewer correction triggers `SendTaskSuccess`.

---

## 9. Future Architecture — Full Production HITL

The current implementation is a Phase 1 POC. If the project moves to a full production deployment, the following enhancements are recommended.

### 9.1 Per-Field Step Function States

Instead of a single `WaitForHuman` state for all fields, the state machine should have one wait state per flagged field, connected in sequence. This eliminates the token-sharing problem entirely.

```
StartAt: DetermineReviewFields
  │
  ┌──────────────────────────────────────────────────────┐
  │  WaitForHuman_total_amount  (waitForTaskToken)       │
  └──────────────────────────┬───────────────────────────┘
                             │ token_1 resolved
  ┌──────────────────────────┴───────────────────────────┐
  │  WaitForHuman_total_fees    (waitForTaskToken)       │
  └──────────────────────────┬───────────────────────────┘
                             │ token_2 resolved
  ┌──────────────────────────┴───────────────────────────┐
  │  WaitForHuman_total_txns    (waitForTaskToken)       │
  └──────────────────────────┬───────────────────────────┘
                             │ all fields resolved
  End
```

### 9.2 Dynamic State Machine Generation

For a variable number of review fields (especially when extending to Visa/Mastercard interchange fields), the state machine definition should be generated programmatically at runtime based on the `review_fields` list, rather than using a fixed single-state definition.

### 9.3 Extension to Visa and Mastercard Interchange Fields

When HITL is extended to interchange program fee fields, the following changes are needed:

- Extend the `per_entity_metrics` structure to include interchange program fields (e.g., `program_code`, `percent_rate`, `flat_rate` per detected program).
- Update the `review_fields` detection logic in `document_ingest()` to include these new fields.
- Extend the `FIELDS` list in the Streamlit frontend (`3_Extracted_with_llm_judge.py`) to render rows for interchange fields.
- The DynamoDB schema requires **no changes** — audit records and task token records use the same key pattern regardless of field name.
- The `hitl_main_pipeline.py` functions (`build_hitl_popup_payload`, `apply_hitl_resolution`) are already field-agnostic and will work without modification.

### 9.4 Additional Production Recommendations

| Area | Recommendation |
|---|---|
| Step Function Timeout | Add `HeartbeatSeconds` and `TimeoutSeconds` to the `WaitForHuman` state to prevent executions from staying open indefinitely if a reviewer never responds. |
| Reviewer Authentication | Replace `reviewer_id` with a real identity (e.g., Cognito or IAM-based) so the audit trail is reliable. |
| Notifications | Add an SNS or SES notification step after the Step Function starts, alerting the reviewer team that a document needs attention. |
| SLA Tracking | Store `review_started_at` and `review_completed_at` timestamps to enable SLA monitoring on how long reviews take. |
| HITL Bypass Option | Add an admin-level bypass that can mark all fields as `HUMAN_ACCEPTED` and resume the Step Function without individual review, for trusted documents. |
| `hitl_resolve_handler.py` | Remove or archive this file. It is a leftover prototype and its presence may cause confusion. The active resolution path is the FastAPI POST route. |
| Error Handling | Add `SendTaskFailure` calls in the catch blocks of `apply_hitl_resolution()` so Step Functions transitions to a `FAILED` state if an unrecoverable error occurs during resolution. |

---

## 10. Summary

### ✅ What Is Working (Phase 1)

- Step Function starts correctly when judge flags fields as `REVIEW` or `INCORRECT`
- `hitl_wait_stub` Lambda stores the `task_token` in DynamoDB
- Frontend shows **Review** button for flagged fields
- Popup returns candidate values with evaluation context
- First reviewer correction saves value, updates PEM metadata, writes audit record
- Step Function resumes (`SendTaskSuccess`) after first field resolution
- DynamoDB schema supports full audit history and version history

### ⚠️ Known Limitation

When multiple fields are flagged, only the first **Review** button triggers a full HITL flow (including Step Function resumption). Subsequent review buttons only update the database. This is caused by the `task_token` being deleted after the first field resolution, and the `remaining_reviews` logic not correctly accounting for fields that have a value but are still labelled `REVIEW` by the judge.

**Fix:** Update `remaining_reviews` to check judge label and `source`, not just value emptiness.

### Files Active in HITL (Phase 1)

| File | Status |
|---|---|
| `main.py` | ✅ Active — API entry point, Step Function trigger |
| `hitl_main_pipeline.py` | ✅ Active — Core HITL business logic |
| `hitl_wait_stub.py` | ✅ Active — Lambda: stores task token |
| `dynamodb_all.py` | ✅ Active — All DynamoDB reads and writes |
| `3_Extracted_with_llm_judge.py` | ✅ Active — Streamlit UI: Review button and modal |
| `terrafrom_main.txt` | ✅ Active — All AWS infrastructure definitions |
| `hitl_resolve_handler.py` | ❌ Not active — early prototype, can be archived |
| `hitl_main.py` | ❌ Not active — superseded by `hitl_main_pipeline.py`, can be archived |

---

*Document prepared for the Phoenix Statement Reporting project — Worldpay.*
*For questions or updates, contact the development team.*
