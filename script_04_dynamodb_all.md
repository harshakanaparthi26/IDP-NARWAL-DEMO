# SCRIPT 4 — dynamodb_all.py
### "The DynamoDB Layer — Data Pipeline Connections"

---

## OPENING

"Now let's look at `dynamodb_all.py`. This file is the single source of truth for all DynamoDB access across the entire system. Every pipeline touches it — but I'm only going to cover the parts that are directly connected to the data pipeline we just walked through.

The rest of this file — evaluation records, HITL feedback, rates, metrics — those are owned by other parts of the team and will be covered separately."

---

## CONFIG & CONNECTION

```python
AWS_REGION = "us-east-2"
DYNAMODB_TABLE = "wp-phoenix-statement-reporting-table"

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)
```

"One table, one connection, initialized at module load time. Every function in this file reuses this single `table` object — no reconnecting on every call. We use `boto3.resource` rather than `boto3.client` because it gives us a cleaner higher-level Table interface."

---

## PK / SK CONSTANTS — The ones the data pipeline uses

```python
PK_DOC = "DOC#"
SK_META = "META"
```

"These two constants are what the data pipeline works with. `DOC#<doc_id>` is the Partition Key for all document-level records. `META` is the Sort Key that identifies the metadata record for a document. Every function `data_main_pipeline.py` calls uses exactly these two keys — so these constants are the shared contract between the pipeline and the database."

---

## Decimal Conversion

```python
def to_decimal(obj: Any) -> Any:
    if isinstance(obj, float):
        return Decimal(str(obj))
    ...

def clean_decimals(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    ...
```

"DynamoDB doesn't support Python `float` — it uses `Decimal`. So every write goes through `to_decimal` and every read comes back through `clean_decimals`. Both are recursive so they handle nested dicts and lists automatically. We use `Decimal(str(obj))` rather than `Decimal(obj)` directly to avoid floating-point precision errors. This runs under every single function in this file."

---

## write_meta_start — Called by data_main_pipeline.py first

```python
def write_meta_start(doc_id, doc_name, s3_raw_key, industry, volume_tier, pricing_type, region):
    item = {
        "PK": f"{PK_DOC}{doc_id}",
        "SK": SK_META,
        "status": "PROCESSING",
        "created_at": datetime.utcnow().isoformat(),
        ...
    }
    table.put_item(Item=to_decimal(item))
```

"This is the very first DynamoDB call in the entire pipeline — `data_main_pipeline.py` calls this before it even touches Textract. It creates the META record with `status = PROCESSING` so the document is tracked in the system from the moment it arrives. We also write GSI keys here so documents can be queried by date — but the core purpose is to mark the document as in-progress immediately."

---

## update_meta_s3_raw_key — Called right after Textract returns

```python
def update_meta_s3_raw_key(doc_id: str, s3_raw_key: str):
    table.update_item(
        Key={"PK": f"{PK_DOC}{doc_id}", "SK": SK_META},
        UpdateExpression="SET s3_raw_key = :k",
        ExpressionAttributeValues={":k": s3_raw_key},
    )
```

"Once `textract.py` finishes and returns the `s3_raw_key`, `data_main_pipeline.py` immediately calls this — before running Comprehend. It patches just that one field onto the existing META record. We do it this early because other pipelines need the S3 key as soon as it exists — we don't wait until the end."

---

## write_meta_complete — Called by data_main_pipeline.py last

```python
def write_meta_complete(doc_id, s3_text_key, index_name, char_count, table_count):
    table.update_item(
        Key={"PK": f"{PK_DOC}{doc_id}", "SK": SK_META},
        UpdateExpression="SET #s = :status, completed_at = :c, s3_text_key = :t, ...",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues=to_decimal({":status": "COMPLETE", ...}),
    )
```

"This is the last DynamoDB call in the data pipeline — called by `data_main_pipeline.py` after both Textract and Comprehend have finished and the redacted text is saved to S3. It flips `status` from `PROCESSING` to `COMPLETE` and records the final S3 key, char count, and table count.

We use `update_item` not `put_item` — so we're merging these fields into the existing record, not overwriting it. `status` needs the `#s` alias because it's a reserved word in DynamoDB's expression syntax — a common gotcha you'll hit if you try to use it directly."

---

## CLOSING

"So those are the three functions from `dynamodb_all.py` that the data pipeline directly calls — `write_meta_start` at the very beginning, `update_meta_s3_raw_key` right after Textract, and `write_meta_complete` at the very end. Together they bookend the entire pipeline and keep the document's state in sync throughout."

---
