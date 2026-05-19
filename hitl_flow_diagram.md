# HITL Pipeline — Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              already covered in previous sessions                   │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Document ingestion                             │   │
│   │       Textract → Comprehend → Evaluation                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │  LLM judge flags REVIEW / INCORRECT
                              ▼
         ┌────────────────────────────────────────┐
         │       main.py  ·  /document/ingest     │   ← Script 01
         │  Collects review_fields                │
         │  → starts Step Function execution      │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │          AWS Step Functions            │   ← Terraform KT
         │  WaitForHuman state                    │
         │  waitForTaskToken → execution pauses ⏸ │
         └────────────────────────────────────────┘
                              │
 Step                         │  invokes Lambda + injects token
 Function         ┌───────────┘
 paused ⏸ ╌╌╌╌╌╌╌│╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
           ╌╌╌╌╌╌╌│╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
                  ▼
         ┌────────────────────────────────────────┐
         │      Lambda  ·  hitl_wait_stub         │   ← Script 02
         │  Receives task token from event        │
         │  Writes token to DynamoDB → returns    │
         └────────────────────────────────────────┘
                              │
                              │  writes HITL#doc_id / SK: TASK
                              ▼
         ┌────────────────────────────────────────┐
         │              DynamoDB                  │   ← Script 04
         │     PK: HITL#doc_id  ·  SK: TASK       │
         │     task_token stored, waiting         │
         └────────────────────────────────────────┘
                              │
                              │  reviewer opens UI, clicks Review button
                              ▼
         ┌────────────────────────────────────────┐
         │         Streamlit frontend             │   ← Script 05
         │  Shows flagged fields + Review button  │
         │  Popup: candidates + judge reasoning   │
         └────────────────────────────────────────┘
                              │
                              │  GET /hitl/{doc_id}/field/{field}
                              │  POST /hitl/{doc_id}/field/{field}
                              ▼
         ┌────────────────────────────────────────┐
         │       main.py  ·  HITL routes          │   ← Script 01
         │  GET → builds popup payload            │
         │  POST → applies correction             │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │        hitl_main_pipeline.py           │   ← Script 03
         │  Persists corrected value              │
         │  Writes audit record                   │
         │  Checks: any fields still empty?       │
         └────────────────────────────────────────┘
                              │
                              │  calls update_separate_prompt_entities
                              │  calls update_hitl_feedback
                              ▼
         ┌────────────────────────────────────────┐
         │           dynamodb_all.py              │   ← Script 04
         │  Targeted nested-map writes            │
         │  Atomic initialize-and-populate        │
         └────────────────────────────────────────┘
                              │
                              │  all fields resolved → SendTaskSuccess
                              ▼
         ┌────────────────────────────────────────┐
         │        Step Function resumes           │   ← Terraform KT
         │  Token deleted from DynamoDB           │
         │  Execution completes  ✓                │
         └────────────────────────────────────────┘


─────────────────────────────────────────────────────
  Presentation order
─────────────────────────────────────────────────────
  Script 00   Overview + this flow diagram
  Script 01   main.py          (trigger + HITL routes)
  Script 02   hitl_wait_stub   (wait Lambda)
  Script 03   hitl_main_pipeline  (core logic)
  Script 04   dynamodb_all     (HITL storage functions)
  Script 05   Streamlit frontend  (reviewer UI)
              hitl_resolve_handler  (brief mention in Script 02)
              Terraform             (separate KT session)
─────────────────────────────────────────────────────
  Legend
─────────────────────────────────────────────────────
  ░░ Gray      Already covered (ingestion / storage)
  ██ Purple    FastAPI / pipeline logic
  ██ Teal      AWS (Step Functions / Lambda)
  ██ Coral     Frontend (Streamlit)
─────────────────────────────────────────────────────
```
