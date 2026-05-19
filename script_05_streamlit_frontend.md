# Script 05 — `3_Extracted_with_llm_judge.py` (Streamlit Frontend)

---

## Opening

Last file — `3_Extracted_with_llm_judge.py`. This is the Streamlit page that the human reviewer actually sees and interacts with. It's the other end of the two API routes we covered in `main.py` — every GET and POST we walked through earlier gets triggered from here. We're going to keep this high level, focusing on the main logic blocks and how they connect to the backend.

---

## Setup and Initial Load

```python
doc_id = ss.get("doc_id")

if not doc_id:
    st.info("No document selected. Please upload a document first.")
    st.stop()

with st.spinner("Loading evaluation results..."):
    record = get_extraction(doc_id)
```

The page requires an active `doc_id` in session state — that's set upstream when a document is uploaded and processed. No `doc_id`, we stop immediately. Once we have one, we call `get_extraction` directly — same function from `dynamodb_all.py` that `hitl_main_pipeline.py` also uses. The frontend reads from DynamoDB directly through the shared storage layer, which avoids an extra API round trip just for the initial page load.

---

## Local Session State

```python
if "ui_entities" not in ss:
    ss["ui_entities"] = dict(entities)

if "hitl_modal" not in ss:
    ss["hitl_modal"] = {
        "open": False,
        "field": None,
        "payload": None,
    }
```

Two pieces of local state. `ui_entities` is a local copy of the extracted entities — this is what the table displays, so when a reviewer submits a correction it shows up immediately without needing to re-read DynamoDB. `hitl_modal` is a simple open/closed flag that tracks whether the review popup is showing and which field it's for.

---

## The Entity Table

```python
FIELDS = [
    "total_amount",
    "total_transactions_count",
    "total_fees",
]

for field in FIELDS:
    val = ss["ui_entities"].get(field)
    judge = get_judge(field)
    verdict = judge.get("label", "-")

    col_entity, col_value, col_verdict = st.columns([2, 3, 2])

    with col_value:
        is_corrected = f"_corrected_{field}" in ss["ui_entities"]

        if is_corrected:
            st.write(ss["ui_entities"][f"_corrected_{field}"])
        elif needs_review(judge):
            if st.button("🔵 Review", key=f"review_{field}"):
                popup = fetch_hitl_popup(field)
                ss["hitl_modal"] = {
                    "open": True,
                    "field": field,
                    "payload": popup,
                }
                st.rerun()
        else:
            st.write(val if val is not None else "-")
```

The table loops over the three tracked fields and renders three columns — field name, value, judge verdict. In the value column the logic has three branches. If the field was already corrected this session, show the corrected value. If the judge flagged it and it hasn't been corrected yet, show the blue Review button. Otherwise show the raw extracted value.

When the reviewer clicks Review, we call `fetch_hitl_popup` — that's a `requests.get` to `GET /hitl/{doc_id}/field/{field}` on our FastAPI backend, the route we covered in `main.py`. The response goes into the modal state and we call `st.rerun()` to re-render the page with the popup now open. That's the standard Streamlit pattern for triggering conditional UI changes.

---

## The Review Popup

```python
if ss["hitl_modal"]["open"]:
    field = ss["hitl_modal"]["field"]
    payload = ss["hitl_modal"]["payload"]

    candidates = payload.get("candidates", [])
    ctx = payload.get("evaluation_context", {})

    st.markdown(f"### Entity: `{field}`")

    with st.expander("Why review was required"):
        st.json(ctx)

    choice = st.radio("Select the correct value:", candidates, key=f"radio_{field}")
    manual_val = st.text_input("Manual value", key=f"manual_{field}") if choice == "manual_input" else None
    final_val = manual_val if manual_val else choice
```

When the modal is open, the popup renders below the table. It shows the field name, an expandable section with the judge's evaluation context — label, score, evidence, rationale — and then a radio button list of the candidate values that came back from the backend. If the reviewer selects the `manual_input` sentinel, a text field appears. Otherwise they just click the right candidate.

```python
    if st.button("✅ Save"):
        res = submit_hitl_value(field, final_val)

        if res.get("ok"):
            ss["ui_entities"][field] = final_val
            ss["ui_entities"][f"_corrected_{field}"] = final_val

            ss["hitl_modal"] = {"open": False, "field": None, "payload": None}

            st.success("Value saved.")
            st.rerun()
```

When Save is clicked, we call `submit_hitl_value` — that's a `requests.post` to `POST /hitl/{doc_id}/field/{field}`, the second HITL route from `main.py`, which feeds into `apply_hitl_resolution` in `hitl_main_pipeline.py`. If the response comes back ok, we update the local session state so the corrected value shows up in the table immediately, close the modal, and rerun. From the reviewer's perspective the correction is reflected instantly.

---

## Closing

And that's the full HITL pipeline end to end. A document gets ingested, the LLM judge flags uncertain fields, `main.py` starts the Step Function, the wait Lambda stores the token, the reviewer corrects fields through this Streamlit page, `hitl_main_pipeline.py` persists everything and checks completion, and when all fields are resolved the Step Function gets woken up and the pipeline finishes.

The infrastructure tying all of this together — the Step Function state machine, the Lambda deployments, IAM roles, everything — is covered in a separate Terraform KT session.

---
