# SCRIPT 6 — How It All Connects
### "The Full Data Pipeline Flow — Start to Finish"

---

## OPENING

"Before we dive into any individual file, I want to give you the full picture first — how all five files connect to each other and how data flows through them end to end. Because if you understand the flow upfront, every file we look at after this will immediately make sense in context.

So let me trace the journey of a single merchant statement PDF from the moment it arrives at the API, all the way to the final response."

---

## STEP 1 — Entry Point: main.py

"Everything starts in `main.py` at the `/document/ingest` POST endpoint.

The ISV uploads a PDF through the UI. FastAPI receives it as an `UploadFile` object, and we immediately call `await file.read()` to get the raw bytes. We also parse the `context` form field — that's where industry, tier, pricing type, and region come from.

From those inputs, we either use the `doc_id` the caller provided, or we generate one by running the filename through `_sanitize_identifier`. That `doc_id` is the single key that ties every record together across DynamoDB, S3, and all five pipelines.

Then `main.py` initializes the `MetricsManager` and makes its first pipeline call:"

```
main.py  →  run_data_main_pipeline(file_bytes, filename, doc_id, ...)
```

---

## STEP 2 — Into data_main_pipeline.py

"Control passes to `data_main_pipeline.py`. This is the orchestrator for the data layer.

The first thing it does is call `write_meta_start` from `dynamodb_all.py` — that writes a `DOC#<doc_id> / META` record to DynamoDB with `status = PROCESSING`. The document now exists in the system.

Then it calls into `textract.py`:"

```
data_main_pipeline.py  →  textract.process_document(file_obj, filename)
```

---

## STEP 3 — Into textract.py

"Inside `textract.py`, three things happen in sequence:

First, `upload_to_s3` takes the raw file bytes and pushes the PDF to S3 under the raw prefix. It returns the `s3_raw_key`.

Second, `run_textract` starts an async Textract job pointing at that S3 key, polls until it completes, and paginates through all result pages.

Third, `parse_blocks` converts the Textract block output into two things — a plain text string and a list of DataFrames for any tables detected.

Those outputs, plus the `s3_raw_key`, are returned back up as a dict:"

```
textract.py  →  returns { status, text, tables, s3_raw_key, s3_text_key }
             →  back to data_main_pipeline.py
```

---

## STEP 4 — Back in data_main_pipeline.py: S3 key persisted

"As soon as `data_main_pipeline.py` gets that `s3_raw_key` back from Textract, it immediately calls `update_meta_s3_raw_key` from `dynamodb_all.py`. This patches the META record in DynamoDB with the real S3 key.

This happens before Comprehend runs — not at the end — because the line item extraction pipeline and other downstream services need that S3 key to be available as soon as possible.

Then it calls into `comprehend.py`:"

```
data_main_pipeline.py  →  comprehend.redact_pii(raw_text)
```

---

## STEP 5 — Into comprehend.py

"Inside `comprehend.py`, the raw text is chunked into 4000-byte pieces. Each chunk is sent to Amazon Comprehend's `detect_pii_entities` API. The entity offsets are translated back to full-document positions, and `_apply_redaction` replaces every detected PII span with a `[[REDACTED:TYPE]]` token.

The redacted text is then saved to S3 via `save_redacted`, which returns an `s3_redacted_key`.

Both the redacted text and the key are returned back up:"

```
comprehend.py  →  returns redacted_text
              →  save_redacted() → s3_redacted_key
              →  back to data_main_pipeline.py
```

---

## STEP 6 — data_main_pipeline.py closes out

"Back in `data_main_pipeline.py`, we now have everything we need. We call `write_meta_complete` from `dynamodb_all.py` — this updates the META record with `status = COMPLETE`, the `s3_text_key` pointing to the redacted file, and the `char_count` and `table_count`.

Then the full data pipeline result is returned back to `main.py`:"

```
data_main_pipeline.py  →  returns {
                              raw_text,
                              redacted_text,
                              tables,
                              s3_raw_key,
                              s3_redacted_key
                           }
                       →  back to main.py
```

---

## STEP 7 — main.py continues the chain

"Back in `main.py`, the redacted text flows into the remaining pipelines in sequence:

- The **embedding pipeline** takes the redacted text, creates vector embeddings, and stores them in OpenSearch. It returns an `index_name`.
- The **retriever pipeline** queries OpenSearch using that index, retrieves the top-k relevant chunks, and runs LLM extraction to produce `separate_prompt_entities` — the structured fields like total fees and total volume.
- The **evaluation pipeline** scores each extracted field and writes the evaluation record to DynamoDB via `dynamodb_all.write_evaluation_record`.
- If any fields are flagged as `REVIEW` or `INCORRECT`, `main.py` triggers a Step Functions execution to start the HITL workflow.
- If industry and tier context were provided, the **benchmark pipeline** runs and compares the extraction against TSG data.

The final response goes back to the frontend with everything assembled."

---

## THE FULL FLOW — Visual Summary

"So if you map it out linearly, the data flows like this:"

```
[ISV uploads PDF]
        ↓
   main.py  (/document/ingest)
        ↓  passes file_bytes + doc_id
   data_main_pipeline.py
        ↓  write_meta_start → dynamodb_all.py  (status = PROCESSING)
        ↓
   textract.py
        ↓  upload PDF → S3
        ↓  run async Textract job
        ↓  parse blocks → raw_text + tables
        ↓  save outputs → S3
        ↓  returns s3_raw_key
        ↓
   data_main_pipeline.py
        ↓  update_meta_s3_raw_key → dynamodb_all.py
        ↓
   comprehend.py
        ↓  chunk text → Comprehend API
        ↓  apply redaction → redacted_text
        ↓  save redacted → S3
        ↓  returns redacted_text + s3_redacted_key
        ↓
   data_main_pipeline.py
        ↓  write_meta_complete → dynamodb_all.py  (status = COMPLETE)
        ↓  returns full data output
        ↓
   main.py
        ↓  embedding pipeline → OpenSearch
        ↓  retriever pipeline → LLM extraction → separate_entities
        ↓  evaluation pipeline → judge scores → dynamodb_all.py
        ↓  [HITL trigger if review_fields exist] → Step Functions
        ↓  benchmark pipeline (if context provided)
        ↓
   [Final JSON response → Frontend]
```

---

## CLOSING / TRANSITION INTO FILE WALKTHROUGH

"So before we dive in — a few things worth anchoring:

`dynamodb_all.py` isn't really a step in the pipeline — it's a layer that runs underneath all of them. Every file touches it. It's the system's memory.

`textract.py` and `comprehend.py` are pure processors — they take an input, do one job, and return an output. They don't know about each other.

`data_main_pipeline.py` is the one that sequences them and keeps DynamoDB in sync at every step.

And `main.py` is the entry point — it receives the real-world request, kicks off the data pipeline, and then chains all the other pipelines on top of the result.

Now that we have the full picture, let's zoom into each file one by one — starting with `textract.py`, which is where the pipeline does its very first piece of real work: getting text out of the PDF."

---
