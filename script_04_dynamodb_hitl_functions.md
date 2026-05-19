# Script 04 — `dynamodb_all.py` (HITL Functions)

---

## Opening

Now let's look at the two DynamoDB functions that `hitl_main_pipeline.py` calls into for its storage operations. You've already seen both of these being called — `update_separate_prompt_entities` when we persist a corrected value, and `update_hitl_feedback` when we write the audit trail to the evaluation record. Now let's understand exactly how they work under the hood.

Quick context: `dynamodb_all.py` is the unified DynamoDB access layer for the entire system. The table setup, PK/SK schema, Decimal conversion helpers, and the META write functions were all covered in an earlier session. I'm only covering the two HITL-specific functions here.

---

## `update_separate_prompt_entities`

```python
def update_separate_prompt_entities(
    doc_id: str,
    updates: Dict[str, Any],
):
    if not updates:
        return

    update_expr = []
    expr_names = {}
    expr_values = {}

    for k, v in updates.items():
        name_key = f"#k_{k}"
        value_key = f":v_{k}"

        update_expr.append(f"separate_prompt_entities.{name_key} = {value_key}")
        expr_names[name_key] = k
        expr_values[value_key] = to_decimal(v)

    table.update_item(
        Key={
            "PK": f"{PK_DOC}{doc_id}",
            "SK": SK_EXTRACTION,
        },
        UpdateExpression="SET " + ", ".join(update_expr),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )
```

This function is called from `apply_hitl_resolution` in `hitl_main_pipeline.py` whenever a human submits a corrected value. Its job is to write that correction back into the existing extraction record in DynamoDB without touching anything else stored there.

The extraction record has a nested map called `separate_prompt_entities` — that holds all the extracted field values and their judge metrics. When a reviewer corrects one field, we do not want to overwrite the entire map. That would mean reading the full record, modifying it in memory, and writing the whole thing back — which introduces race conditions if multiple reviewers are working at the same time, and is wasteful.

Instead we use DynamoDB's `update_item` with a `SET` expression that targets nested map keys directly. The pattern `separate_prompt_entities.#k_total_amount = :v_total_amount` tells DynamoDB to set just that one nested attribute without touching anything else in the map. Surgical, atomic, safe.

The `#k_` prefix on the expression attribute names is important. DynamoDB has a list of reserved words — things like `status`, `name`, `type`. If your field name happens to match one of those, your update will fail with a cryptic error. Using the `#` prefix escapes that. We apply it to every key as a blanket rule rather than checking case by case, because it's always safe and costs nothing.

The `to_decimal` call on each value is because DynamoDB uses the `Decimal` type internally instead of Python floats. That conversion function was covered in the earlier DynamoDB session.

And the guard at the top — `if not updates: return` — is just a safety check. If called with an empty dictionary we bail immediately rather than building a malformed update expression that would fail at the API level.

---

## `update_hitl_feedback`

```python
def update_hitl_feedback(doc_id: str, hitl_payload: Dict):
    update_expr = []
    expr_values = {}
    expr_names = {}

    # Ensure parent map exists
    update_expr.append("hitl = if_not_exists(hitl, :empty_map)")
    expr_values[":empty_map"] = {}

    for k, v in hitl_payload.items():
        name_key = f"#k_{k}"
        value_key = f":v_{k}"

        update_expr.append(f"hitl.{name_key} = {value_key}")
        expr_names[name_key] = k
        expr_values[value_key] = to_decimal(v)

    table.update_item(
        Key={
            "PK": f"{PK_EVAL}{doc_id}",
            "SK": SK_EVAL_LATEST,
        },
        UpdateExpression="SET " + ", ".join(update_expr),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=expr_values,
    )
```

This function is also called from `apply_hitl_resolution`. While `update_separate_prompt_entities` writes the corrected value into the extraction record, `update_hitl_feedback` writes the audit event into the evaluation record. These are two separate DynamoDB items — the extraction record lives under `DOC#<doc_id>` and the evaluation record lives under `EVAL#<doc_id>`. So these two writes are going to completely different places in the table.

The evaluation record has a `hitl` map nested inside it that stores things like who reviewed the field, what they corrected, the accuracy score after correction, and when it happened. The `if_not_exists` expression on the first line is the critical detail here. The `hitl` map might not exist yet — this could be the very first correction for this document. If you try to set `hitl.field = value` and the `hitl` map doesn't exist yet, DynamoDB will throw an error because you can't set a nested attribute on a map that isn't there.

So we use `if_not_exists(hitl, :empty_map)` — which tells DynamoDB: if this attribute doesn't exist yet, initialize it as an empty map first, then apply the rest of the SET expressions. This is a DynamoDB conditional write idiom that lets you safely initialize and populate a nested map in a single atomic operation. Much cleaner than doing a read-check-write sequence.

The rest of the function follows exactly the same pattern as `update_separate_prompt_entities` — expression attribute names to escape reserved words, `to_decimal` for type safety, dynamic SET expression built in a loop.

Notice the target key: `PK_EVAL` and `SK_EVAL_LATEST`. This writes to the latest evaluation pointer record, not a versioned snapshot. The versioned history is written separately by `write_evaluation_record` which was covered in the earlier DynamoDB session. HITL feedback always goes onto the latest record because it reflects the current human-reviewed state of the document.

---

## Closing

Those are the two DynamoDB functions the HITL pipeline owns. One writes corrected values back into the extraction record with a targeted nested-map update, the other writes the audit trail into the evaluation record with an atomic initialize-and-populate pattern.

Next up is the Streamlit frontend — `3_Extracted_with_llm_judge.py` — where we'll see exactly what the reviewer sees and how these API calls get made from the UI side.

---
