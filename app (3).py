# app.py
"""
Streamlit UI — Statement Intelligence Platform.

All AWS / ML logic lives in pipeline.py.
This file only handles layout, session state, and displaying results.
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
    get_industries,
    get_volume_tiers,
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
.k  { color:#6b7280; }
.v  { color:#111827; font-weight:600; }
.hero { background:linear-gradient(90deg,#0b1220,#111827,#0b1220); border:1px solid #111827; border-radius:16px; padding:18px; color:white; margin-bottom:14px; }
.hero h1 { font-size:1.8rem; font-weight:750; margin:0; }
.hero p  { margin:6px 0 0 0; color:#cbd5e1; font-size:0.98rem; }
.step-header { background:#0b1220; color:white; padding:10px 12px; border-radius:12px; margin:18px 0 10px 0; border:1px solid #111827; }
.step-header h3 { font-size:1.05rem; margin:0; font-weight:700; }
.answer { background:#fff; border:1px solid #16213a; border-radius:12px; padding:14px; }
.bench-box { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 16px; margin-top:8px; }
.bench-box-warn { background:#fefce8; border:1px solid #fde68a; border-radius:10px; padding:14px 16px; margin-top:8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "current_step":    1,
    "pdf_processed":   False,
    "docs":            [],
    "qa_history":      [],
    "bench_df":        None,
    "bench_industries": [],
    "bench_industry":  None,
    "bench_tiers":     [],      # volume tiers for selected industry
    "bench_tier":      None,    # selected volume tier
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
    # preserve benchmark data across resets
    ss.bench_df         = st.session_state.get("bench_df")
    ss.bench_industries = st.session_state.get("bench_industries", [])
    ss.bench_industry   = st.session_state.get("bench_industry")
    ss.bench_tiers      = st.session_state.get("bench_tiers", [])
    ss.bench_tier       = st.session_state.get("bench_tier")


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-load benchmarks once
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_INDUSTRIES = [
    "Business Services", "Public Administration", "Digital Goods",
    "Eating & Drinking Places", "Healthcare", "Higher Risk", "Other",
    "Personal Services", "Entertainment & Recreation", "B2B", "Construction",
    "Educational, Non-Profits, Public Services, & Interest Groups",
    "Farming & Agriculture", "Finance, Insurance, & Real Estate",
    "Manufacturing & Mining", "Airline, Hospitality, & Car Rental",
    "Grocery & Petrol", "Retail", "Transportation, Communication, & Utilities",
    "Other - Not Classified Elsewhere",
]

if ss.bench_df is None:
    df_bench = load_benchmarks()
    if df_bench is not None:
        ss.bench_df         = df_bench
        ss.bench_industries = get_industries(df_bench)
    else:
        ss.bench_industries = FALLBACK_INDUSTRIES
        st.sidebar.warning("Benchmarks CSV not loaded. Using fallback industry list.")

    if ss.bench_industries and ss.bench_industry not in ss.bench_industries:
        ss.bench_industry = ss.bench_industries[0]


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
        weight = "700"    if done else "500"
        st.sidebar.markdown(
            f"<div style='display:flex;gap:8px;align-items:center;padding:4px 0;'>"
            f"<span style='color:{color};font-size:12px;'>{icon}</span>"
            f"<span style='color:{color};font-weight:{weight};font-size:12px;'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_rag_config():
    st.sidebar.markdown('<div class="sidebar-title">Live Configuration</div>', unsafe_allow_html=True)
    for k, v in [("Chunk size", ss.chunk_size), ("Overlap", ss.chunk_overlap),
                 ("Top-K", ss.top_k), ("Top-N", ss.top_n), ("MMR λ", ss.mmr_lambda)]:
        st.sidebar.markdown(
            f"<div class='kv'><div class='k'>{k}</div><div class='v'>{v}</div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar layout
# ═══════════════════════════════════════════════════════════════════════════════

_render_logo()
_render_progress(ss.current_step)
_render_rag_config()

# ── Benchmark selectors ───────────────────────────────────────────────────────
with st.sidebar.expander("Benchmark Settings", expanded=True):
    st.markdown("**Select industry and volume tier to compare effective rate against the TSG benchmark.**")

    # Industry picker
    industry_idx = (
        ss.bench_industries.index(ss.bench_industry)
        if ss.bench_industry in ss.bench_industries else 0
    )
    selected_industry = st.selectbox(
        "Merchant Industry (MERCHANT_GROUP)",
        options=ss.bench_industries,
        index=industry_idx,
        key="sidebar_industry",
        help="Select the merchant's industry category.",
    )

    # When industry changes, refresh the volume tier list
    if selected_industry != ss.bench_industry:
        ss.bench_industry = selected_industry
        ss.bench_tiers    = get_volume_tiers(ss.bench_df, selected_industry) if ss.bench_df is not None else []
        ss.bench_tier     = ss.bench_tiers[0] if ss.bench_tiers else None

    # Populate tiers on first load
    if not ss.bench_tiers and ss.bench_df is not None and ss.bench_industry:
        ss.bench_tiers = get_volume_tiers(ss.bench_df, ss.bench_industry)
        ss.bench_tier  = ss.bench_tiers[0] if ss.bench_tiers else None

    # Volume tier picker (depends on selected industry)
    if ss.bench_tiers:
        tier_idx = (
            ss.bench_tiers.index(ss.bench_tier)
            if ss.bench_tier in ss.bench_tiers else 0
        )
        selected_tier = st.selectbox(
            "Volume Tier (VOLUME_TIER_TSG)",
            options=ss.bench_tiers,
            index=tier_idx,
            key="sidebar_tier",
            help="Select the merchant's annual processing volume tier.",
        )
        ss.bench_tier = selected_tier
    else:
        st.caption("No volume tiers available for the selected industry.")
        ss.bench_tier = None

    # Show current selection summary
    if ss.bench_industry and ss.bench_tier:
        st.success(f"✓ {ss.bench_industry}  |  {ss.bench_tier}")
        print("TSG DATA Load")
    else:
        st.info("Select both industry and volume tier to enable benchmark comparison.")

# ── Advanced RAG settings ─────────────────────────────────────────────────────
with st.sidebar.expander("Advanced RAG Settings", expanded=False):
    new_chunk_size    = st.slider("Chunk size",             200,  3000, ss.chunk_size,               step=100)
    new_chunk_overlap = st.slider("Chunk overlap",            0,   800, ss.chunk_overlap,             step=50)
    new_top_k         = st.slider("Top-K retrieved",          4,    50, ss.top_k,                    step=1)
    new_top_n         = st.slider("Top-N to LLM (≤ Top-K)",   1, new_top_k, min(ss.top_n, new_top_k), step=1)
    new_mmr_lambda    = st.slider("MMR λ",                  0.0,   1.0, float(ss.mmr_lambda),        step=0.05)

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

    # Warn if benchmark not fully configured
    if not ss.bench_industry or not ss.bench_tier:
        st.warning("⚠️ Select both **Merchant Industry** and **Volume Tier** in the sidebar before processing to enable benchmark comparison.")

    colA, colB = st.columns([2, 1])
    with colA:
        st.caption("Pipeline: Textract → PII Redaction → RAG Index → Entity Extraction → Benchmark Comparison")
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
        progress_bar    = st.progress(0.0)
        status_text     = st.empty()
        total           = len(uploaded_files)
        processed       = []

        for idx, f in enumerate(uploaded_files, start=1):
            def on_status(msg: str, _t=status_text):
                _t.info(msg)

            progress_bar.progress((idx - 1) / total)

            result = run_document_pipeline(
                file            = f,
                filename        = f.name,
                bench_df        = ss.bench_df,
                merchant_group  = ss.bench_industry,
                volume_tier_tsg = ss.bench_tier,
                on_status       = on_status,
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

if ss.pdf_processed:
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Process new document(s)", type="secondary", width="stretch"):
            reset()
            st.rerun()
    with c2:
        st.caption("Tip: Change the industry or volume tier in the sidebar and re-run to update benchmark comparison.")


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
                st.warning(f"Could not load redacted text: {e}")

    with tab_raw:
        st.caption("Raw text is not persisted (privacy). Enable storage explicitly if needed.")

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
# Step 3 — Q&A + Results
# ═══════════════════════════════════════════════════════════════════════════════

if ss.pdf_processed and ss.docs:
    st.markdown('<div class="step-header"><h3>Step 3 — Q&A & Results</h3></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Q&A", "Extraction Comparison", "IDP Results", "Benchmark Comparison"])

    # ── Tab 1: Q&A ────────────────────────────────────────────────────────────
    with tabs[0]:
        qa_opts    = {f"{d['doc_name']} ({d['doc_id'][:6]})": d for d in ss.docs}
        qa_pick    = st.selectbox("Q&A on document:", list(qa_opts.keys()))
        active_doc = qa_opts.get(qa_pick)

        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the total amount processed?",
        )
        ask_btn = st.button("Ask", type="primary")

        if ask_btn and question and active_doc:
            with st.spinner("Processing..."):
                try:
                    ans = ask(
                        question   = question,
                        doc_id     = active_doc["doc_id"],
                        index_name = active_doc["index_name"],
                    )
                    ss.qa_history.append({
                        "question":    question,
                        "answer":      ans["answer"],
                        "chunks_used": ans["chunks_used"],
                        "timestamp":   datetime.now().strftime("%H:%M:%S"),
                        "judge":       ans.get("judge_result", {}),
                        "cache_hit":   ans.get("cache_hit", False),
                    })
                    st.markdown("#### Answer")
                    st.markdown(f"<div class='answer'>{ans['answer']}</div>", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Chunks used",    ans["chunks_used"])
                    c2.metric("Judge verdict",  ans["judge_result"].get("label", "—"))

                    with st.expander(f"Top chunks (used {ans['chunks_used']})", expanded=False):
                        for i, chunk in enumerate(ans["top_k_chunks"][:3], 1):
                            st.markdown(f"**Chunk {i} (id={chunk.get('chunk_id')}):**")
                            txt = chunk.get("text", "")
                            st.code((txt[:320] + "…") if len(txt) > 320 else txt)
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Extraction comparison (single vs separate) ─────────────────────
    with tabs[1]:
        st.markdown("### Single Prompt vs Separate Prompts")
        st.caption("Both methods extract the same 3 entities. Separate prompts use independent retrieval + judge per entity.")

        for doc in ss.docs:
            st.markdown(f"#### {doc['doc_name']}")
            single   = doc.get("single_prompt_entities",   {})
            separate = doc.get("separate_prompt_entities", {})

            rows = []
            for entity in ["total_amount", "total_transactions_count", "total_fees"]:
                sv = single.get(entity)
                pv = separate.get(entity)
                rows.append({
                    "Entity":           entity,
                    "Single Prompt":    sv,
                    "Separate Prompts": pv,
                    "Match":            "✓" if sv == pv else "✗",
                })
            st.table(pd.DataFrame(rows))

            eff = compute_effective_rate(separate)
            st.metric(
                "Effective Rate (raw) — from separate prompts",
                f"{eff['effective_rate_raw']:.6f}" if eff["effective_rate_raw"] else "—",
            )
            if eff.get("notes"):
                st.caption(f"Note: {eff['notes']}")
            st.markdown("---")

    # ── Tab 3: Full IDP table ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### IDP Entity Extraction Results")
        rows = []
        for doc in ss.docs:
            sep = doc.get("separate_prompt_entities", {})
            sng = doc.get("single_prompt_entities",   {})
            eff = compute_effective_rate(sep)
            for entity in ["total_amount", "total_transactions_count", "total_fees"]:
                metrics = sep.get("per_entity_metrics", {}).get(entity, {})
                rows.append({
                    "document":          doc["doc_name"],
                    "entity":            entity,
                    "separate_prompts":  sep.get(entity),
                    "single_prompt":     sng.get(entity),
                    "judge_verdict":     (metrics.get("judge") or {}).get("label"),
                    "chunks_used":       metrics.get("chunks_used"),
                })
            rows.append({
                "document": doc["doc_name"], "entity": "effective_rate_raw",
                "separate_prompts": eff.get("effective_rate_raw"), "single_prompt": None,
                "judge_verdict": None, "chunks_used": None,
            })
            comp = doc.get("benchmark_comparison")
            if comp:
                for key, val in [
                    ("benchmark_raw",          comp.get("benchmark_raw")),
                    ("delta_vs_benchmark_raw", comp.get("delta_vs_benchmark_raw")),
                    ("volume_tier_tsg",        comp.get("volume_tier_tsg")),
                ]:
                    rows.append({
                        "document": doc["doc_name"], "entity": key,
                        "separate_prompts": val, "single_prompt": None,
                        "judge_verdict": None, "chunks_used": None,
                    })

        if rows:
            df_out = pd.DataFrame(rows)
            st.dataframe(df_out, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download (JSON)",
                    data=json.dumps(rows, indent=2, default=str).encode("utf-8"),
                    file_name=f"idp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            with c2:
                st.download_button(
                    "Download (CSV)",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name=f"idp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            st.info("No IDP results yet.")

    # ── Tab 4: Benchmark comparison ───────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Benchmark Comparison")

        # Show current selection
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Industry:** {ss.bench_industry or 'Not selected'}")
        with col_b:
            st.info(f"**Volume Tier:** {ss.bench_tier or 'Not selected'}")

        if not ss.bench_industry or not ss.bench_tier:
            st.warning("Select both **Merchant Industry** and **Volume Tier** in the sidebar to see benchmark comparison.")
        else:
            for doc in ss.docs:
                st.markdown(f"#### {doc['doc_name']}")
                sep  = doc.get("separate_prompt_entities", {})
                comp = doc.get("benchmark_comparison")

                # If benchmark was run at process time with different selections,
                # re-compute live with current sidebar selections
                if (
                    comp is None or
                    comp.get("merchant_group") != ss.bench_industry or
                    comp.get("volume_tier_tsg") != ss.bench_tier
                ):
                    if ss.bench_df is not None:
                        comp = compare_with_benchmark(
                            entities        = sep,
                            merchant_group  = ss.bench_industry,
                            volume_tier_tsg = ss.bench_tier,
                            bench_df        = ss.bench_df,
                        )
                        doc["benchmark_comparison"] = comp  # cache result

                if comp and comp.get("benchmark_raw") is not None:
                    er    = comp.get("effective_rate_raw")
                    bench = comp.get("benchmark_raw")
                    delta = comp.get("delta_vs_benchmark_raw")

                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Effective Rate (raw)",
                        f"{er:.6f}" if er is not None else "—",
                    )
                    c2.metric(
                        "Benchmark (raw)",
                        f"{bench:.6f}" if bench is not None else "—",
                    )
                    c3.metric(
                        "Δ Effective − Benchmark",
                        f"{delta:+.6f}" if delta is not None else "—",
                        delta_color="inverse",   # red if positive (above benchmark = worse)
                    )

                    # Contextual verdict
                    if delta is not None:
                        if delta > 0:
                            st.markdown(
                                f"<div class='bench-box-warn'>⚠️ Effective rate is <b>{delta:+.6f}</b> above the benchmark "
                                f"for <b>{ss.bench_industry}</b> / <b>{ss.bench_tier}</b>.</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='bench-box'>✅ Effective rate is <b>{abs(delta):.6f}</b> below the benchmark "
                                f"for <b>{ss.bench_industry}</b> / <b>{ss.bench_tier}</b>.</div>",
                                unsafe_allow_html=True,
                            )

                    with st.expander("Benchmark details", expanded=False):
                        st.json({
                            "merchant_group":         comp.get("merchant_group"),
                            "volume_tier_tsg":        comp.get("volume_tier_tsg"),
                            "volume_tier_id":         comp.get("volume_tier_id"),
                            "effective_rate_raw":     er,
                            "benchmark_raw":          bench,
                            "delta_vs_benchmark_raw": delta,
                            "notes":                  comp.get("notes", ""),
                        })
                else:
                    st.error(
                        f"No benchmark found for **{ss.bench_industry}** / **{ss.bench_tier}**. "
                        "Try a different industry or volume tier."
                    )
                    if comp and comp.get("notes"):
                        st.caption(comp["notes"])

                st.markdown("---")


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
