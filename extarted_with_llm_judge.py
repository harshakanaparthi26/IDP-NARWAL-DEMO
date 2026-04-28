import streamlit as st
import pandas as pd
import requests
import sys
from pathlib import Path
from backend.src.storage.dynamodb_all import get_extraction
from backend.src.services import settings
from utils.state import ss, ensure_defaults

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SERVICES_DIR = REPO_ROOT / "backend" / "src" / "services"
if str(SERVICES_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICES_DIR))

ensure_defaults()

st.set_page_config(
    page_title="Extracted Entities (HITL Enabled)",
    page_icon="📄",
    layout="wide"
)

st.header("📄 Extracted Entities with LLM Judge")

# ------------------------------------------------------------
# Require active document
# ------------------------------------------------------------
doc_id = ss.get("doc_id")

if not doc_id:
    st.info("No document selected. Please upload a document first.")
    st.stop()

# ------------------------------------------------------------
# Load evaluation data from DynamoDB
# ------------------------------------------------------------
with st.spinner("Loading evaluation results..."):
    record = get_extraction(doc_id)

if not record:
    st.error("No evaluation record found.")
    st.stop()

entities = record.get("separate_prompt_entities", {}) or {}
per_metrics = entities.get("per_entity_metrics", {}) or {}

API_URL = settings.BACKEND_URL

FIELDS = [
    "total_amount",
    "total_transactions_count",
    "total_fees"
]

# ------------------------------------------------------------
# LOCAL UI OVERRIDES
# ------------------------------------------------------------
if "ui_entities" not in ss:
    ss["ui_entities"] = dict(entities)

if "hitl_modal" not in ss:
    ss["hitl_modal"] = {
        "open": False,
        "field": None,
        "payload": None
    }

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def get_judge(field):
    return (per_metrics.get(field) or {}).get("judge", {})


def needs_review(judge):
    return judge.get("label") in ("REVIEW", "INCORRECT")


def fetch_hitl_popup(field):
    return requests.get(
        f"{API_URL}/hitl/{doc_id}/field/{field}"
    ).json()


def submit_hitl_value(field, value):
    payload = {
        "corrected_value": value,
        "reviewer_id": ss.get("user_id", "frontend_user"),
        "task_token": ss["hitl_modal"].get("task_token"),
    }
    return requests.post(
        f"{API_URL}/hitl/{doc_id}/field/{field}",
        json=payload
    ).json()


# ------------------------------------------------------------
# Unified inline table
# ------------------------------------------------------------
st.subheader("📄 Extracted Entities")

# Header row
hcol1, hcol2, hcol3 = st.columns([2, 3, 2])
hcol1.markdown("**Entity**")
hcol2.markdown("**Value**")
hcol3.markdown("**Verdict**")

st.divider()

for field in FIELDS:
    val = ss["ui_entities"].get(field)
    judge = get_judge(field)
    verdict = judge.get("label", "-")

    col_entity, col_value, col_verdict = st.columns([2, 3, 2])

    with col_entity:
        st.write(f"**{field}**")

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
                    "payload": popup
                }
                st.rerun()
        else:
            st.write(val if val is not None else "-")

    with col_verdict:
        is_corrected = f"_corrected_{field}" in ss["ui_entities"]
        if is_corrected:
            st.write("✅ CORRECTED")
        else:
            st.write(verdict)

    st.divider()

# ------------------------------------------------------------
# HITL Modal (renders below the table when open)
# ------------------------------------------------------------
if ss["hitl_modal"]["open"]:
    st.markdown("---")
    st.subheader("🔍 Review & Correct Value")

    field = ss["hitl_modal"]["field"]
    payload = ss["hitl_modal"]["payload"]

    if not payload or not payload.get("ok"):
        st.error("Unable to load review data.")
    else:
        candidates = payload.get("candidates", [])
        ctx = payload.get("evaluation_context", {})

        st.markdown(f"### Entity: `{field}`")

        with st.expander("Why review was required"):
            st.json(ctx)

        choice = st.radio("Select the correct value:", candidates, key=f"radio_{field}")
        manual_val = st.text_input("Manual value", key=f"manual_{field}") if choice == "manual_input" else None
        final_val = manual_val if manual_val else choice

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("✅ Save"):
                res = submit_hitl_value(field, final_val)

                if res.get("ok"):
                    # Store corrected value and mark as corrected
                    ss["ui_entities"][field] = final_val
                    ss["ui_entities"][f"_corrected_{field}"] = final_val

                    ss["hitl_modal"] = {
                        "open": False,
                        "field": None,
                        "payload": None
                    }
                    st.success("Value saved.")
                    st.rerun()
                else:
                    st.error(res)

        with col_b:
            if st.button("❌ Cancel"):
                ss["hitl_modal"] = {
                    "open": False,
                    "field": None,
                    "payload": None
                }
                st.rerun()
