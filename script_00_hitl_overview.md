# Script 00 — HITL Pipeline: Overview & Flow

---

## Opening

Before we dive into individual files, let me show you the full picture first — how this entire Human-in-the-Loop pipeline connects, which files are involved, and where AWS services fit in. I'll walk you through this flow diagram and then we'll go file by file.

---

## The Problem We're Solving

So the product processes merchant statements — it extracts key financial fields like total fees, total volume, total transaction count. We use an LLM to do that extraction, and we also run a judge model on top that scores how confident it is in each value. Sometimes the judge isn't confident. Sometimes the document is ambiguous. And in a financial product, you cannot silently pass a wrong number downstream. So we built a Human-in-the-Loop layer — if the model flags a field, we pause the pipeline, a human reviews and corrects it, and only then does the pipeline continue.

---

## Walking the Flow

*(point at the top of the diagram)*

At the top you can see the ingestion layer — Textract, Comprehend, and the evaluation pipeline. Those were all covered in earlier sessions. What we care about here is what happens right after evaluation finishes.

*(point at "Document ingestion" box)*

The evaluation pipeline runs the LLM judge over every extracted field and writes judge scores into DynamoDB — labels like REVIEW or INCORRECT for anything it's uncertain about.

*(point at "main.py · /document/ingest" box)*

Right after that, still inside the ingest endpoint in main.py, we loop over those judge results, collect any field that needs human review, and if there's anything in that list — we kick off an AWS Step Function execution.

*(point at "AWS Step Functions" box)*

The Step Function has one state — WaitForHuman — and it uses a special pattern called waitForTaskToken. This means it invokes a Lambda but does not wait for it to return normally. Instead it generates a unique task token, passes it into the Lambda, and then pauses — it literally suspends and waits indefinitely until something external tells it to resume.

*(point at "Lambda · hitl_wait_stub" box)*

That Lambda's entire job is to take that task token and store it in DynamoDB. That's it. One write. It returns immediately. The Step Function is now paused and the token is safely stored on the backend.

*(point at the dashed bracket on the left side)*

And it stays paused — could be a minute, could be an hour — until a human acts.

*(point at "DynamoDB" box)*

The token sits here under a HITL key for this document. Nothing moves until we retrieve it later.

*(point at "Streamlit frontend" box)*

Meanwhile the reviewer opens the frontend — that's our Streamlit page. It shows the extracted entities in a table. For any field the judge flagged, instead of showing a value, it shows a blue Review button. The reviewer clicks it, a popup appears with candidate values and the judge's reasoning, they pick the right value or type one in, and hit Save.

*(point at "main.py · HITL routes" box)*

That Save action hits two API routes in main.py — a GET to build the popup, and a POST to submit the correction.

*(point at "hitl_main_pipeline.py" box)*

Both routes delegate immediately to hitl_main_pipeline.py — that's where the real logic lives. It persists the corrected value, writes an audit record, and then checks whether all flagged fields now have values.

*(point at "Step Function resumes" box)*

If everything is resolved, it retrieves that task token from DynamoDB, calls SendTaskSuccess, and the Step Function wakes up and completes. The token gets deleted. Pipeline done.

---

## The Files — One Line Each

Just so you know what we're about to cover, in this order:

`dynamodb_all.py` — the two HITL-specific storage functions that everything else calls into.

`hitl_wait_stub.py` — the Lambda that stores the task token when the Step Function starts.

`hitl_resolve_handler.py` — a second Lambda from an earlier architectural approach, covered briefly for context.

`hitl_main_pipeline.py` — the core business logic. This is the main file.

`main.py` — just the two HITL API routes.

`3_Extracted_with_llm_judge.py` — the Streamlit review UI, high level.

And the Terraform — that provisions the Step Function, the Lambdas, IAM roles, all the infrastructure. Covered in a separate Terraform KT session, not going into it here.

---

## Closing

That's the full map. Let's now go file by file, starting with the DynamoDB layer.

---
