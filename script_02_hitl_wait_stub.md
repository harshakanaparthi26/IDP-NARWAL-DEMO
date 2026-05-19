# Script 02 — `hitl_wait_stub.py`

---

## Opening

Alright, so `main.py` just started the Step Function execution. The very next thing that happens is the Step Function invokes this Lambda — `hitl_wait_stub.py`. This is a short file, under 30 lines, but it does something architecturally critical. Understanding it means understanding one of the most useful patterns in AWS Step Functions — the `waitForTaskToken` callback pattern. Let's walk through it.

Quick side note before we dive in — there's a second Lambda in this system called `hitl_resolve_handler.py`. That was part of an earlier design where resolution happened directly through a Lambda rather than through our FastAPI backend. We moved away from that approach, so the active resolution path runs through `hitl_main_pipeline.py` which we'll cover next. I'm mentioning it now so it doesn't catch you off guard — we're not covering it as its own deep dive.

---

## Imports and Setup

```python
import json
import boto3
from datetime import datetime
from backend.src.services import settings

dynamodb = boto3.resource("dynamodb", region_name=settings.AWS_REGION)
table = dynamodb.Table(settings.DYNAMODB_TABLE)
```

Standard setup. boto3 DynamoDB resource pointed at our table using the settings module for region and table name. Same pattern used everywhere in this codebase.

---

## The Handler

```python
def handler(event, context):
    doc_id = event["doc_id"]
    task_token = event["task_token"]

    # STORE token for later resolution
    table.put_item(
        Item={
            "PK": f"HITL#{doc_id}",
            "SK": "TASK",
            "task_token": task_token,
            "created_at": datetime.utcnow().isoformat(),
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "doc_id": doc_id,
            "field": event.get("field"),
            "task_token": task_token
        })
    }
```

Let me explain how this Lambda gets invoked, because the invocation pattern is what makes this interesting.

The Terraform configuration — covered in a separate Terraform KT session — defines the Step Function state machine with a single state called `WaitForHuman`. That state uses the resource type `arn:aws:states:::lambda:invoke.waitForTaskToken`. That `.waitForTaskToken` suffix is everything. It tells Step Functions: invoke this Lambda, but do not advance the state machine based on the Lambda's return value. Instead, generate a unique task token, inject it into the Lambda's event payload, and then pause the entire execution indefinitely — it will wait forever until something externally calls `SendTaskSuccess` with that token.

So by the time this handler runs, the event already has two things in it: the `doc_id` that `main.py` passed in when it called `start_execution`, and a `task_token` that was automatically generated and injected by Step Functions itself.

The Lambda's job is one thing: write that token to DynamoDB under `PK = HITL#<doc_id>` and `SK = TASK`. That's the key that `hitl_main_pipeline.py` will look up later when the human finishes reviewing and we're ready to resume the execution.

You might ask — why store the token at all? Why not just pass it through to the frontend and have the frontend send it back when they submit? The answer is reliability and security. We don't want a task token floating around in API responses or on the frontend longer than necessary. Storing it in DynamoDB keeps it on the backend. The frontend never needs to know about task tokens — it just calls our API with a `doc_id` and a field name, and our backend handles the Step Functions resume internally.

One more thing worth calling out: the Lambda returns a `statusCode: 200` response, but that response goes nowhere useful. Because of the `waitForTaskToken` pattern, Step Functions has already suspended — it's not waiting for this Lambda to return. The return value is effectively ignored by the state machine. It's there as good practice for CloudWatch logging, but it doesn't drive anything.

---

## Why `put_item` and Not `update_item`

Small but intentional design choice: we use `put_item` rather than `update_item`. That means if a task token record already exists for this `doc_id`, it gets completely overwritten. That's deliberate — if the same document gets reprocessed and a new Step Function execution starts, we want the latest token to win. A stale token from a previous run that can never be resolved is worse than overwriting it.

---

## Closing

That's `hitl_wait_stub.py`. The Step Function is now paused. The token is in DynamoDB. A reviewer could take a minute or an hour — the execution will wait.

Now let's look at `hitl_main_pipeline.py` — this is the core of the HITL system, where all the real logic lives: building the review popup, applying corrections, checking completion, and waking the Step Function back up.

---
