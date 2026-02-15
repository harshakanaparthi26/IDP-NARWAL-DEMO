# app.py
"""
Streamlit UI — Statement Intelligence Platform.

All AWS / ML logic lives in pipeline.py.
This file only handles:
  - Page layout and session state
  - Calling pipeline functions
  - Displaying results (st.* calls)
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import settings
from pipeline import (
    run_document_pipeline,
    ask,
    load_benchmarks,
    compute_effective_rate,
    compare_with_benchmark,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Worldpay | Statement Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: Inter, system-ui, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }
section[data-testid="stSidebar"] { background:#fafafa; border-right:1px solid #e5e7eb; }
.sidebar-title { font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; font-weight:700; margin-bottom:8px; }
.kv { display:flex; justify-content:space-between; font-size:12px; padding:6px 0; border-bottom:1px dashed #f3f4f6; }
.k { color:#6b7280; } .v { color:#111827; font-weight:600; }
.hero { background:linear-gradient(90deg,#0b1220,#111827,#0b1220); border:1px solid #111827; border-radius:16px; padding:18px; color:white; margin-bottom:14px; }
.hero h1 { font-size:1.8rem; font-weight:750; margin:0; }
.hero p  { margin:6px 0 0 0; color:#cbd5e1; font-size:0.98rem; }
.step-header { background:#0b1220; color:white; padding:10px 12px; border-radius:12px; margin:18px 0 10px 0; border:1px solid #111827; }
.step-header h3 { font-size:1.05rem; margin:0; font-weight:700; }
.answer { background:#fff; border:1px solid #16213a; border-radius:12px; padding:14px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state helpers
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "current_step":    1,
    "pdf_processed":   False,
    "docs":            [],        # List of pipeline result dicts, one per document
    "qa_history":      [],
    "bench_df":        None,
    "bench_industries": [],
    "bench_industry":  None,
    # RAG params
    "chunk_size":    settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
    "top_k":         settings.TOP_K,
    "top_n":         settings.TOP_N,
    "mmr_lambda":    settings.MMR_LAMBDA,
}

ss = st.session_state
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)


def reset():
    for k, v in DEFAULTS.items():
        ss[k] = v
    ss.bench_df         = None
    ss.bench_industries = []


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_logo():
    logo = Path("assets/worldpay-logo.png")
    if logo.exists():
        st.sidebar.image(Image.open(logo), width="stretch")
    else:
        st.sidebar.markdown(
            "<div style='font-weight:800;font-size:18px;color:#111827;'>worldpay</div>",
            unsafe_allow_html=True,
        )


def _render_progress(step: int):
    steps = [(1, "Upload PDF"), (2, "Textract"), (3, "RAG Index"), (4, "Ready")]
    st.sidebar.markdown('<div class="sidebar-title">Processing Status</div>', unsafe_allow_html=True)
    for i, label in steps:
        done   = step >= i
        icon   = "●" if done else "○"
        color  = "#111827" if done else "#9ca3af"
        weight = "700" if done else "500"
        st.sidebar.markdown(
            f"<div style='display:flex;gap:8px;align-items:center;padding:4px 0;'>"
            f"<span style='color:{color};font-size:12px;'>{icon}</span>"
            f"<span style='color:{color};font-weight:{weight};font-size:12px;'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_config():
    st.sidebar.markdown('<div class="sidebar-title">Live Configuration</div>', unsafe_allow_html=True)
    for k, v in [("Chunk size", ss.chunk_size), ("Overlap", ss.chunk_overlap),
                 ("Top-K", ss.top_k), ("Top-N", ss.top_n), ("MMR λ", ss.mmr_lambda)]:
        st.sidebar.markdown(
            f"<div class='kv'><div class='k'>{k}</div><div class='v'>{v}</div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-load benchmarks (once)
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_INDUSTRIES = [
    "Business Services", "Digital Goods", "Eating & Drinking Places",
    "Healthcare", "Retail", "Other", "B2B", "Finance, Insurance, & Real Estate",
]

if ss.bench_df is None:
    df_bench = load_benchmarks()
    if df_bench is not None:
        ss.bench_df = df_bench
        inds = sorted(df_bench["MERCHANT_GROUP"].dropna().astype(str).str.strip().unique().tolist())
        ss.bench_industries = inds or FALLBACK_INDUSTRIES
    else:
        ss.bench_industries = FALLBACK_INDUSTRIES
        st.sidebar.warning("Benchmarks CSV not loaded. Using fallback industry list.")
    ss.bench_industry = ss.bench_industries[0] if ss.bench_industries else None


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar layout
# ═══════════════════════════════════════════════════════════════════════════════

_render_logo()
_render_progress(ss.current_step)
_render_config()

with st.sidebar.expander("Merchant Industry", expanded=True):
    ss.bench_industry = st.selectbox(
        "Select Merchant Industry (MERCHANT_GROUP)",
        options=ss.bench_industries,
        index=max(ss.bench_industries.index(ss.bench_industry) if ss.bench_industry in ss.bench_industries else 0, 0),
        help="Used to pull the TSG benchmark for the right industry + volume tier.",
    )
    if ss.bench_df is not None:
        print("TSG DATA Load")

with st.sidebar.expander("Advanced RAG Settings", expanded=False):
    new_chunk_size    = st.slider("Chunk size",             200,  3000, ss.chunk_size,    step=100)
    new_chunk_overlap = st.slider("Chunk overlap",            0,   800, ss.chunk_overlap,  step=50)
    new_top_k         = st.slider("Top-K retrieved",          4,    50, ss.top_k,          step=1)
    new_top_n         = st.slider("Top-N to LLM (≤ Top-K)",  1, new_top_k, min(ss.top_n, new_top_k), step=1)
    new_mmr_lambda    = st.slider("MMR λ (relevance ↔ diversity)", 0.0, 1.0, float(ss.mmr_lambda), step=0.05)
    apply_rebuild     = st.button("Apply & rebuild index", type="primary", width="stretch")

# Sync sliders to session
if any([new_chunk_size != ss.chunk_size, new_chunk_overlap != ss.chunk_overlap,
        new_top_k != ss.top_k, new_top_n != ss.top_n, float(new_mmr_lambda) != float(ss.mmr_lambda)]):
    ss.chunk_size    = new_chunk_size
    ss.chunk_overlap = new_chunk_overlap
    ss.top_k         = new_top_k
    ss.top_n         = new_top_n
    ss.mmr_lambda    = float(new_mmr_lambda)


# ═══════════════════════════════════════════════════════════════════════════════
# Hero header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>Statement Intelligence Platform</h1>
  <p>Extract, index, and query merchant statements using hybrid retrieval and LLM answers.</p>
  <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
    <span style="font-size:11px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);color:#e5e7eb;">AWS Textract</span>
    <span style="font-size:11px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);color:#e5e7eb;">Hybrid RAG</span>
    <span style="font-size:11px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);color:#e5e7eb;">BM25 + OpenSearch</span>
    <span style="font-size:11px;padding:4px 10px;border-radius:999px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);color:#e5e7eb;">Claude 3</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Upload + Process
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="step-header"><h3>Step 1 — Upload document(s)</h3></div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    f"Upload up to {settings.MAX_DOCS} PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload statement PDFs to run IDP extraction and enable Q&A.",
)

if uploaded_files:
    if len(uploaded_files) > settings.MAX_DOCS:
        st.warning(f"You uploaded {len(uploaded_files)} files. Only the first {settings.MAX_DOCS} will be processed.")
        uploaded_files = uploaded_files[: settings.MAX_DOCS]

    with st.container():
        for f in uploaded_files:
            st.markdown(f"• **{f.name}** — {len(f.getvalue()):,} bytes")

    colA, colB = st.columns([2, 1])
    with colA:
        st.caption("Processing runs sequentially: Textract → PII Redaction → RAG Index → Entity Extraction.")
    with colB:
        run_clicked = st.button(
            "Run IDP extraction",
            type="primary",
            width="stretch",
            disabled=ss.pdf_processed and len(ss.docs) > 0,
        )

    if run_clicked:
        reset()
        ss.current_step = 2
        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        total        = len(uploaded_files)
        processed    = []

        for idx, f in enumerate(uploaded_files, start=1):

            def on_status(msg: str, _text=status_text):
                _text.info(msg)

            progress_bar.progress((idx - 1) / total)
            result = run_document_pipeline(
                file=f,
                filename=f.name,
                bench_df=ss.bench_df,
                merchant_group=ss.bench_industry,
                on_status=on_status,
            )

            if result.get("error"):
                st.error(f"Failed: {f.name} — {result['error']}")
            else:
                processed.append(result)
            progress_bar.progress(idx / total)

        ss.docs          = processed
        ss.pdf_processed = len(processed) > 0
        ss.current_step  = 4

        status_text.success("IDP extraction complete. Results are ready.")
        st.success(f"Processed {len(processed)} document(s).")

# Reset button
if ss.pdf_processed:
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Process new document(s)", type="secondary", width="stretch"):
            reset()
            st.rerun()
    with c2:
        st.caption("Tip: For Q&A, the last processed document stays active by default.")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Extracted content preview
# ═══════════════════════════════════════════════════════════════════════════════

if ss.pdf_processed and ss.docs:
    st.markdown('<div class="step-header"><h3>Step 2 — Extracted content</h3></div>', unsafe_allow_html=True)

    import boto3 as _boto3
    _s3 = _boto3.client("s3")

    doc_options = {f"{d['doc_name']} ({d['doc_id'][:6]})": d for d in ss.docs}
    pick_label  = st.selectbox("Select a document to preview:", list(doc_options.keys()))
    picked      = doc_options.get(pick_label)

    tab_red, tab_raw, tab_tbl = st.tabs(["Redacted Text", "Raw Text (note)", "Tables"])

    with tab_red:
        if picked:
            try:
                obj     = _s3.get_object(Bucket=settings.S3_BUCKET, Key=picked["s3_text_key"])
                red_txt = obj["Body"].read().decode("utf-8")
                with st.expander("Redacted Text (preview)", expanded=False):
                    st.text(red_txt[:2000])
                if len(red_txt) > 2000:
                    st.caption(f"… and {len(red_txt) - 2000:,} more characters")
                st.download_button(
                    "Download Redacted Text",
                    data=red_txt.encode("utf-8"),
                    file_name=picked["doc_name"].replace(".pdf", "_redacted.txt"),
                    mime="text/plain",
                )
            except Exception as e:
                st.warning(f"Could not load redacted text from S3: {e}")

    with tab_raw:
        st.caption("Raw text is not persisted (for privacy). Enable storage explicitly if needed.")

    with tab_tbl:
        tables = picked.get("tables", []) if picked else []
        if tables:
            st.info(f"Found {len(tables)} table(s)")
            for i, table in enumerate(tables, 1):
                st.subheader(f"Table {i}")
                try:
                    st.dataframe(table if isinstance(table, pd.DataFrame) else pd.DataFrame(table), use_container_width=True)
                except Exception:
                    st.json(table)
        else:
            st.caption("No tables detected.")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Q&A + IDP Results
# ═══════════════════════════════════════════════════════════════════════════════

if ss.pdf_processed and ss.docs:
    st.markdown('<div class="step-header"><h3>Step 3 — Q&A & Results</h3></div>', unsafe_allow_html=True)

    qa_tabs = st.tabs(["Interactive Q&A", "Extraction Comparison", "IDP Results"])

    # ── Interactive Q&A ───────────────────────────────────────────────────────
    with qa_tabs[0]:
        qa_doc_opts = {f"{d['doc_name']} ({d['doc_id'][:6]})": d for d in ss.docs}
        qa_pick     = st.selectbox("Q&A on document:", list(qa_doc_opts.keys()))
        active_doc  = qa_doc_opts.get(qa_pick)

        question  = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the total amount processed?",
        )
        ask_btn = st.button("Ask", type="primary")

        if ask_btn and question and active_doc:
            with st.spinner("Processing..."):
                try:
                    answer_result = ask(
                        question=question,
                        doc_id=active_doc["doc_id"],
                        index_name=active_doc["index_name"],
                    )
                    ss.qa_history.append({
                        "question":   question,
                        "answer":     answer_result["answer"],
                        "chunks_used": answer_result["chunks_used"],
                        "timestamp":  datetime.now().strftime("%H:%M:%S"),
                        "judge":      answer_result.get("judge_result", {}),
                        "cache_hit":  answer_result.get("cache_hit", False),
                    })
                    st.markdown("#### Answer")
                    st.markdown(f"<div class='answer'>{answer_result['answer']}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Chunks used", answer_result["chunks_used"])
                    c2.metric("Judge verdict", answer_result["judge_result"].get("label", "—"))

                    with st.expander(f"Top chunks (used {answer_result['chunks_used']})", expanded=False):
                        for i, chunk in enumerate(answer_result["top_k_chunks"][:3], 1):
                            st.markdown(f"**Chunk {i} (id={chunk.get('chunk_id')}):**")
                            txt = chunk.get("text", "")
                            st.code((txt[:320] + "…") if len(txt) > 320 else txt)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Extraction comparison (single vs separate) ────────────────────────────
    with qa_tabs[1]:
        st.markdown("### Single Prompt vs Separate Prompts — Entity Extraction Comparison")
        for doc in ss.docs:
            st.markdown(f"#### {doc['doc_name']}")
            single   = doc.get("single_prompt_entities", {})
            separate = doc.get("separate_prompt_entities", {})

            rows = []
            for entity in ["total_amount", "total_transactions_count", "total_fees"]:
                rows.append({
                    "Entity":          entity,
                    "Single Prompt":   single.get(entity),
                    "Separate Prompts": separate.get(entity),
                    "Match":           "✓" if single.get(entity) == separate.get(entity) else "✗",
                })
            st.table(pd.DataFrame(rows))

            # Effective rate for separate (more reliable — per-entity judged)
            eff = compute_effective_rate(separate)
            st.metric("Effective Rate (raw)", f"{eff['effective_rate_raw']:.6f}" if eff["effective_rate_raw"] else "—")

            if doc.get("benchmark_comparison"):
                comp = doc["benchmark_comparison"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Benchmark (raw)",   f"{comp['benchmark_raw']:.6f}"           if comp.get("benchmark_raw") else "—")
                c2.metric("Δ Eff – Bench",     f"{comp['delta_vs_benchmark_raw']:+.6f}" if comp.get("delta_vs_benchmark_raw") is not None else "—")
                c3.metric("Tier",              comp.get("volume_tier_tsg") or "—")
            st.markdown("---")

    # ── IDP Results table ─────────────────────────────────────────────────────
    with qa_tabs[2]:
        st.markdown("### IDP Entity Extraction Results")
        rows = []
        for doc in ss.docs:
            separate = doc.get("separate_prompt_entities", {})
            single   = doc.get("single_prompt_entities", {})
            eff      = compute_effective_rate(separate)
            for entity in ["total_amount", "total_transactions_count", "total_fees"]:
                metrics = separate.get("per_entity_metrics", {}).get(entity, {})
                rows.append({
                    "document_name":   doc["doc_name"],
                    "doc_id":          doc["doc_id"],
                    "entity_key":      entity,
                    "value_separate":  separate.get(entity),
                    "value_single":    single.get(entity),
                    "judge_verdict":   (metrics.get("judge") or {}).get("label"),
                    "chunks_used":     metrics.get("chunks_used"),
                })
            rows.append({
                "document_name":  doc["doc_name"],
                "doc_id":         doc["doc_id"],
                "entity_key":     "effective_rate_raw",
                "value_separate": eff.get("effective_rate_raw"),
                "value_single":   None,
                "judge_verdict":  None,
                "chunks_used":    None,
            })
            if doc.get("benchmark_comparison"):
                comp = doc["benchmark_comparison"]
                for key, val in [
                    ("benchmark_raw",          comp.get("benchmark_raw")),
                    ("delta_vs_benchmark_raw", comp.get("delta_vs_benchmark_raw")),
                    ("matched_volume_tier_tsg", comp.get("volume_tier_tsg")),
                ]:
                    rows.append({
                        "document_name": doc["doc_name"], "doc_id": doc["doc_id"],
                        "entity_key": key, "value_separate": val, "value_single": None,
                        "judge_verdict": None, "chunks_used": None,
                    })

        if rows:
            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download Results (JSON)",
                data=json.dumps(rows, indent=2, default=str).encode("utf-8"),
                file_name=f"idp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
            st.download_button(
                "Download Results (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"idp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No IDP results yet. Upload and process documents in Step 1.")


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A history + export
# ═══════════════════════════════════════════════════════════════════════════════

if ss.qa_history:
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download Q&A History (JSON)",
            data=json.dumps(ss.qa_history, indent=2, default=str),
            file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with c2:
        if st.button("🗑 Clear Q&A History"):
            ss.qa_history = []
            st.rerun()
