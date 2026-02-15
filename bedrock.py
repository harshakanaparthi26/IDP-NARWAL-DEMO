# bedrock.py
"""
Step 4 — Amazon Bedrock (Claude) for extraction, Q&A, and LLM-as-a-judge.

Functions:
    ask_question(question, chunks)                          → answer_str
    extract_entities_single_prompt(chunks)                  → {"total_amount", "total_transactions_count", "total_fees"}
    extract_entities_separate_prompts(chunks_fn)            → same dict, with per-entity judge metrics
    judge_answer(question, context, answer)                 → JudgeResult dict
    run_with_reflection(question, context, max_cycles)      → {"final_answer", "final_judge", "cycles", "cache_hit"}
"""
import json
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import botocore

import settings


# ── Bedrock client ────────────────────────────────────────────────────────────
_bedrock = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)

# ── Cache dir ─────────────────────────────────────────────────────────────────
_cache_dir = Path(settings.JUDGE_CACHE_DIR)
_cache_dir.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Core invoke helper
# ═══════════════════════════════════════════════════════════════════════════════

def _invoke(body: dict, max_retries: int = 8) -> dict:
    """
    Invoke a Bedrock model with exponential-backoff retry on throttling.
    Applies guardrails if enabled in settings.
    Returns the parsed response body dict.
    """
    body_str = json.dumps(body).encode("utf-8")
    kwargs: Dict[str, Any] = {
        "modelId": body.pop("_model_id", settings.BEDROCK_MODEL_QA),
        "body": body_str,
    }
    if settings.ENABLE_GUARDRAIL and settings.BEDROCK_GUARDRAIL_ARN:
        kwargs["guardrailIdentifier"] = settings.BEDROCK_GUARDRAIL_ARN
        kwargs["guardrailVersion"]    = settings.BEDROCK_GUARDRAIL_VERSION

    for attempt in range(max_retries):
        try:
            resp = _bedrock.invoke_model(**kwargs)
            return json.loads(resp["body"].read())
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "Throttling", "TooManyRequestsException"):
                time.sleep(0.5 * (2 ** attempt) + random.random())
                continue
            raise
    raise RuntimeError("Bedrock: max retries exceeded (throttled).")


def _call(system: str, user: str, model_id: str, max_tokens: int = 800, temperature: float = 0.0) -> str:
    """Single Claude call. Returns the text content of the first response block."""
    body = {
        "_model_id": model_id,
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }
    data = _invoke(body)
    return (data.get("content", [{}])[0].get("text") or "").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# JSON helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _normalize_numeric(val: Any) -> Any:
    """Strip currency symbols, commas, parentheses. Return absolute float/int or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return abs(val)
    if isinstance(val, str):
        s = val.strip()
        is_paren = s.startswith("(") and s.endswith(")")
        if is_paren:
            s = s[1:-1]
        s = re.sub(r"[^0-9.\-]", "", s)
        if not s or s in ("-", ".", "-.", ".-"):
            return None
        try:
            return abs(float(s))
        except Exception:
            return None
    return None


def _ctx_from_chunks(chunks: List[Dict]) -> str:
    return "\n\n".join(c["text"] for c in chunks if c.get("text"))


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-a-judge reflection loop
# ═══════════════════════════════════════════════════════════════════════════════

def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.strip().encode()).hexdigest()


def _judge_prompt(question: str, context: str, answer: str) -> str:
    return f"""
You are an impartial evaluator for a RAG QA system.
Output JSON ONLY matching this schema:

{{
  "label": "CORRECT" | "HALLUCINATION" | "INCOMPLETE",
  "rubric": {{
    "groundedness": 1-5,
    "factuality": 1-5,
    "completeness": 1-5,
    "format": 1-5,
    "safety": 1-5
  }},
  "notes": "short critique or empty string"
}}

Label rules:
- CORRECT: fully grounded in context, complete, no errors.
- HALLUCINATION: claims not supported by or contradicting context.
- INCOMPLETE: missing required parts even if grounded.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ===
{answer}

Return JSON ONLY:
""".strip()


def run_with_reflection(
    question: str,
    context: str,
    max_cycles: int = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Generate an answer and iteratively judge + refine it.

    Returns:
        {
          "final_answer": str,
          "final_judge":  dict,
          "best_cycle":   int,
          "cycles":       list,
          "cache_hit":    bool,
        }
    """
    max_cycles = max_cycles or settings.REFLECTION_CYCLES

    # ── Cache lookup ──
    cache_key = _stable_hash(json.dumps({
        "question": question,
        "context_hash": _stable_hash(context),
        "max_cycles": max_cycles,
    }, sort_keys=True))
    cache_file = _cache_dir / f"{cache_key}.json"

    if use_cache and cache_file.exists():
        result = json.loads(cache_file.read_text("utf-8"))
        result["cache_hit"] = True
        print(f"[Bedrock] Judge cache hit")
        return result

    # ── Reflection loop ──
    gen_system = (
        "You are a financial statements assistant. "
        "Answer strictly using ONLY the provided context. "
        "If the answer is not in the context, reply exactly: 'insufficient context'."
    )
    judge_system = "You are a strict evaluator. Output JSON only."

    reflection_history: List[str] = []
    cycles: List[Dict] = []
    halluc_streak = incomplete_streak = 0

    for cycle in range(1, max_cycles + 1):
        # Generate answer
        gen_ctx = "Context:\n" + context
        if reflection_history:
            gen_ctx += "\n\nPrevious critique:\n" + "\n".join(reflection_history)
        answer = _call(gen_system, gen_ctx + f"\n\nQuestion: {question}\nAnswer briefly:", settings.BEDROCK_MODEL_QA)

        # Judge answer
        jp = _judge_prompt(question, context, answer)
        judge_raw = _call(judge_system, jp, settings.BEDROCK_MODEL_JUDGE, max_tokens=400)
        judge_parsed = _safe_json(judge_raw)

        if not judge_parsed or "label" not in judge_parsed:
            judge_parsed = {
                "label": "INCOMPLETE",
                "rubric": {"groundedness": 3, "factuality": 3, "completeness": 2, "format": 1, "safety": 5},
                "notes": f"Judge output unparseable: {judge_raw[:200]}",
            }

        cycles.append({"cycle": cycle, "answer": answer, "judge": judge_parsed})
        label = judge_parsed.get("label", "INCOMPLETE")

        # Streak tracking
        if label == "HALLUCINATION":
            halluc_streak += 1; incomplete_streak = 0
        elif label == "INCOMPLETE":
            incomplete_streak += 1; halluc_streak = 0
        else:
            halluc_streak = incomplete_streak = 0

        # Stop conditions
        if cycle >= 1 and label == "CORRECT":
            break
        if halluc_streak >= 2 or incomplete_streak >= 2:
            break

        reflection_history.append(
            f"Label={label}; scores={judge_parsed.get('rubric',{})}; notes={judge_parsed.get('notes','')}"
        )

    # Pick best cycle by weighted rubric score
    def _score(c: Dict) -> float:
        r = c["judge"].get("rubric", {})
        return (2.0 * r.get("groundedness", 0) + 2.0 * r.get("factuality", 0) +
                2.0 * r.get("completeness", 0) + 0.5 * r.get("format", 0) + 0.5 * r.get("safety", 0))

    best = max(cycles, key=_score)
    result = {
        "final_answer": best["answer"],
        "final_judge":  best["judge"],
        "best_cycle":   best["cycle"],
        "cycles":       cycles,
        "cache_hit":    False,
    }

    if use_cache:
        cache_file.write_text(json.dumps(result, indent=2), "utf-8")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Q&A
# ═══════════════════════════════════════════════════════════════════════════════

def ask_question(question: str, chunks: List[Dict]) -> Dict[str, Any]:
    """
    Answer a question using retrieved chunks as context.
    Runs judge reflection and returns the best answer.

    Returns:
        {
          "answer": str,
          "chunks_used": int,
          "top_k_chunks": list,
          "judge_result": dict,
        }
    """
    context = _ctx_from_chunks(chunks[: settings.TOP_N])
    print(f"[Bedrock] Q&A with {len(chunks[:settings.TOP_N])} chunk(s)")
    result = run_with_reflection(question, context)
    print(f"[Bedrock] Q&A complete — label={result['final_judge'].get('label')} cycle={result['best_cycle']}")
    return {
        "answer":      result["final_answer"],
        "chunks_used": len(chunks[: settings.TOP_N]),
        "top_k_chunks": chunks,
        "judge_result": result["final_judge"],
        "cycles":       result["cycles"],
        "cache_hit":    result["cache_hit"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Entity extraction — METHOD 1: Single prompt (all 3 entities at once)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_entities_single_prompt(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Extract total_amount, total_transactions_count, total_fees
    using ONE combined Bedrock prompt.

    Returns:
        {
          "total_amount": float | None,
          "total_transactions_count": int | None,
          "total_fees": float | None,
          "method": "single_prompt",
          "judge_result": dict,
        }
    """
    context = _ctx_from_chunks(chunks[: settings.TOP_N])
    print("[Bedrock] Extraction — single prompt")

    system = (
        "You extract structured entities from a merchant statement. "
        "Return VALID JSON ONLY. No prose. No markdown."
    )
    user = f"""
Extract these three fields from the context below:
- total_amount: total amount submitted/processed/settled for the period.
- total_transactions_count: total number of transactions/items.
- total_fees: total fees or charges for the period.

Rules:
- Return JSON with exactly these keys: total_amount, total_transactions_count, total_fees.
- If multiple matches for total_amount exist, pick the largest value.
- Return absolute values (no negatives).
- If a value is missing, set it to null. Do not guess.

Context:
{context}
"""

    raw = _call(system, user, settings.BEDROCK_MODEL_EXTRACT, max_tokens=600)
    extracted = _safe_json(raw) or {"total_amount": None, "total_transactions_count": None, "total_fees": None}

    # Judge + correct via reflection
    judge_query = (
        'Validate and correct this JSON. Return JSON ONLY with keys '
        '["total_amount","total_transactions_count","total_fees"]. '
        f'EXTRACTED: {json.dumps(extracted)}'
    )
    judge_result = run_with_reflection(judge_query, context)
    corrected = _safe_json(judge_result["final_answer"]) or {}

    if all(k in corrected for k in ("total_amount", "total_transactions_count", "total_fees")):
        extracted = corrected

    result = {
        "total_amount":             _normalize_numeric(extracted.get("total_amount")),
        "total_transactions_count": extracted.get("total_transactions_count"),
        "total_fees":               _normalize_numeric(extracted.get("total_fees")),
        "method":                   "single_prompt",
        "judge_result":             judge_result["final_judge"],
    }
    print(f"[Bedrock] Single-prompt result: {result}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Entity extraction — METHOD 2: Separate prompts (one per entity + judge)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_one_entity(context: str, entity_key: str, retrieval_query: str, extra_rules: str) -> Tuple[Any, Dict]:
    """
    Extract a single entity with its own prompt + judge reflection.
    Returns (value, judge_result_dict).
    """
    system = (
        f"You extract exactly ONE field ('{entity_key}') from a merchant statement. "
        "Return VALID JSON ONLY (no prose, no markdown)."
    )
    user = f"""
Rules:
- Return JSON with exactly this single key: "{entity_key}".
- The value must be a number when possible; else null.
- Return the absolute value.
- Do not guess. Use ONLY the provided context.
{extra_rules}

Context:
{context}

Extract and return JSON for the single key: {entity_key}.
"""
    raw = _call(system, user, settings.BEDROCK_MODEL_EXTRACT, max_tokens=300)
    candidate = _safe_json(raw) or {entity_key: None}

    # Judge + correct
    judge_query = (
        f'Validate and correct this JSON. Return JSON ONLY with exactly this key: ["{entity_key}"]. '
        f"If value unknown, set null. EXTRACTED: {json.dumps(candidate)}"
    )
    judge_result = run_with_reflection(judge_query, context)
    corrected = _safe_json(judge_result["final_answer"]) or candidate

    if isinstance(corrected, dict) and entity_key in corrected:
        value = corrected[entity_key]
    else:
        value = candidate.get(entity_key)

    return value, judge_result


def extract_entities_separate_prompts(retrieve_fn) -> Dict[str, Any]:
    """
    Extract total_amount, total_transactions_count, total_fees each with their
    own retrieval query, prompt, and judge reflection cycle.

    Args:
        retrieve_fn: callable(query: str) → List[chunk_dicts]
                     (wraps rag.retrieve so bedrock.py stays import-free from rag.py)

    Returns:
        {
          "total_amount":             float | None,
          "total_transactions_count": int | None,
          "total_fees":               float | None,
          "method":                   "separate_prompts",
          "per_entity_metrics": {
              "total_amount":             {"judge": dict, "chunks_used": int},
              "total_transactions_count": {"judge": dict, "chunks_used": int},
              "total_fees":               {"judge": dict, "chunks_used": int},
          }
        }
    """
    print("[Bedrock] Extraction — separate prompts (3 independent retrievals)")

    # ── total_amount ──────────────────────────────────────────────────────────
    q_amount = (
        "Find the total amount submitted/processed/settled or total sales/volume amount for the period. "
        "Look for 'total amount submitted', 'total volume', 'total amount processed', 'statement total'."
    )
    amt_chunks = retrieve_fn(q_amount)
    amt_ctx    = _ctx_from_chunks(amt_chunks[: settings.TOP_N])
    amt_rules  = (
        "- If multiple candidates exist, select the LARGEST plausible currency value.\n"
        "- EXCLUDE values clearly from fee sections.\n"
        "- Strip symbols and commas; return a NUMBER, not a string.\n"
    )
    amt_val, amt_judge = _extract_one_entity(amt_ctx, "total_amount", q_amount, amt_rules)
    amt_val = _normalize_numeric(amt_val)
    print(f"[Bedrock]   total_amount = {amt_val}")

    # ── total_transactions_count ──────────────────────────────────────────────
    q_txn = (
        "Find the total number of transactions/items/orders processed. "
        "Look for 'item count', 'number of items', 'total transactions', 'transaction count'."
    )
    txn_chunks = retrieve_fn(q_txn)
    txn_ctx    = _ctx_from_chunks(txn_chunks[: settings.TOP_N])
    txn_rules  = "- Return an integer when possible; choose the larger plausible total if multiple exist.\n"
    txn_val, txn_judge = _extract_one_entity(txn_ctx, "total_transactions_count", q_txn, txn_rules)
    # Coerce to int
    if isinstance(txn_val, float) and txn_val.is_integer():
        txn_val = int(txn_val)
    elif isinstance(txn_val, str):
        try:
            txn_val = int(float(txn_val))
        except Exception:
            txn_val = _normalize_numeric(txn_val)
    print(f"[Bedrock]   total_transactions_count = {txn_val}")

    # ── total_fees ────────────────────────────────────────────────────────────
    q_fees = (
        "Find the total fees or total charges and fees for the period. "
        "Look for 'total fees', 'total charges and fees', 'total processing fees'."
    )
    fee_chunks = retrieve_fn(q_fees)
    fee_ctx    = _ctx_from_chunks(fee_chunks[: settings.TOP_N])
    fee_rules  = "- Choose the grand total for all fees; exclude refunds unless explicitly included.\n"
    fee_val, fee_judge = _extract_one_entity(fee_ctx, "total_fees", q_fees, fee_rules)
    fee_val = _normalize_numeric(fee_val)
    print(f"[Bedrock]   total_fees = {fee_val}")

    return {
        "total_amount":             amt_val,
        "total_transactions_count": txn_val,
        "total_fees":               fee_val,
        "method":                   "separate_prompts",
        "per_entity_metrics": {
            "total_amount":             {"judge": amt_judge["final_judge"], "chunks_used": min(settings.TOP_N, len(amt_chunks))},
            "total_transactions_count": {"judge": txn_judge["final_judge"], "chunks_used": min(settings.TOP_N, len(txn_chunks))},
            "total_fees":               {"judge": fee_judge["final_judge"], "chunks_used": min(settings.TOP_N, len(fee_chunks))},
        },
    }
