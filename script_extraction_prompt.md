# Script — `_EXTRACTION_PROMPT` Walkthrough

---

## PART 1 — High Level: Why Prompts Like This Exist

So before we dive into the actual prompt line by line, I want to spend a few minutes on the bigger picture — why we even need a prompt this detailed, and what happens when you don't have it.

When you're working with an LLM, the model doesn't come with business context. It doesn't know what an interchange fee is versus a service fee versus a chargeback. It doesn't know that Visa and Mastercard fees are a specific subset of everything on a merchant statement. All it knows is language — it predicts what the most statistically likely response looks like based on its training data. And that's exactly the problem.

There are four failure modes we ran into in this pipeline specifically. Not theoretical failures — things we actually saw in production. And every single part of this prompt exists because of one of those failures.

**Failure one — extracting the wrong fees.** When we first just said "extract the fees," the model extracted everything. Service fees, equipment fees, PCI compliance fees, chargeback fees — all of it. Because from the model's perspective, those are all fees. It had no reason to discriminate. That's not a bug, that's actually the model doing exactly what you told it to do.

**Failure two — hallucinating values.** This one was subtle and dangerous. The model would sometimes look at a row that had a transaction count and a total dollar amount, and it would calculate the per-transaction rate. It would back-solve. The number it returned looked completely real, it was mathematically correct — but it was never in the document. For financial data, that's a serious problem. An invented fee that looks real is worse than a missing fee.

**Failure three — skipping rows.** Merchant statements can have 80, 90, sometimes over 100 interchange line items. The model has a natural summarization bias baked in from training — it was rewarded for concise outputs. So it would silently compress repetitive rows. You'd get 40 rows back from a document with 80. No error, no warning — just quiet data loss.

**Failure four — inconsistent output format.** Even when the data was right, the format was different every time. Sometimes a markdown table, sometimes JSON, sometimes a paragraph. Column names changed — "Rate" versus "Percent" versus "Fee Percent." Without a strict anchor, the model defaults to whatever pattern it saw most in training data for similar tasks.

Now let's walk through the prompt and see exactly how we solved each of these.

---

## PART 2 — Line by Line Walkthrough

---

### The Role

```
You are an expert payments statement analyst.
```

This single line is called a **system persona** or role framing. And it matters more than it looks. When you give the model a role, you're not just being polite — you're activating a specific cluster of knowledge and behavior in the model. "Expert payments statement analyst" tells the model to approach this with precision, domain specificity, and attention to financial detail. Compare that to no role at all, where the model defaults to a general assistant posture — helpful, verbose, and willing to fill in gaps. We don't want gap-filling here. We want strict extraction. The role sets that tone from the very first token.

---

### The Task Definition

```
TASK:
Extract ALL Visa and Mastercard interchange fee line items.
```

Short and explicit. The word ALL is doing real work here — it's a signal against summarization. And "interchange fee line items" is precise domain language that scopes extraction to the right category. But this alone wasn't enough, which is why we needed everything that follows.

---

### The Two-Section Structure

```
OUTPUT FORMAT (STRICT):
- Output TWO sections:
  # VISA Interchange Fees
  # MASTERCARD Interchange Fees
```

We split Visa and Mastercard into separate sections deliberately. When they were mixed in one table, we saw the model mislabel rows — especially for fees with similar naming patterns across both networks. By forcing two separate sections, we reduce the chance of cross-network label confusion. The word STRICT in the header is intentional — it's a signal to the model that this format is non-negotiable.

---

### The Field Definitions

```
1. network (VISA, MASTERCARD)
2. description (Program Name as shown in the statement)
3. number
4. amount
5. total
6. Rate (Percent) (e.g., 1.43%)
7. percent / Per-Transaction Fee (e.g., $0.10)
```

This is where we define exactly what we want extracted. Notice field 2 — "Program Name as shown in the statement." That phrase "as shown in the statement" is critical. Without it, the model would normalize or clean up the names. It might see "VS COMM B2B VIRT PMT P1" and decide to expand it to something more readable. We don't want that. We need the exact string as it appears, because downstream systems match on that exact string.

---

### The Important Requirements Block

```
Important requirements:
- Carefully read the ENTIRE text and all tables from beginning to end.
- Search all sections including tables, line items, summaries, footnotes,
  fee schedules, and detailed rate breakdowns.
```

This addresses a specific failure mode we saw — the model would find the first interchange table near the top of the document and stop there. Merchant statements often have fee schedules buried in footnotes or appendices at the end. This instruction forces the model to treat the entire document as a search space, not just the first obvious table.

---

### The Anti-Hallucination Rules

```
- DO NOT calculate, derive, infer, back-solve, estimate, or reverse-engineer
  any rate or fee.
```

This is one of the most important lines in the entire prompt. And notice we didn't just say "don't calculate." We said calculate, derive, infer, back-solve, estimate, reverse-engineer. Every single one of those synonyms is there intentionally. In LLM terms, these are different reasoning paths. The model might understand "don't calculate" as "don't do arithmetic" — but it might still think "inferring" a value from context is fine. By listing every variant of the same behavior, we close each reasoning path explicitly. This is directly a response to seeing hallucinated rates in production.

```
- DO NOT create or hallucinate any fees.
```

A direct, blunt instruction. Sometimes you need to say the thing plainly. "Hallucinate" is a term the model recognizes from its training about its own failure modes — using the word activates the model's awareness of that specific risk.

---

### The Anti-Skip Rules

```
- Capture EVERY relevant Visa interchange fee, even if the document contains
  many rows. Do not skip any row or details between rows.
- DO NOT skip any line item when extracting the details.
- DO NOT summarize or filter. Include every fee line explicitly shown in the text.
```

Three separate instructions saying the same thing. That's not an accident. In a long prompt, instructions given once near the top can lose weight by the time the model is generating row 60 of an 80-row table. Repeating this across multiple phrasings reinforces it at different points in the model's attention. We saw row-skipping disappear after we added this repetition.

---

### The Exact Preservation Rule

```
- Preserve the names and numbers exactly as shown in the statement.
```

Again — "as shown in the statement." This prevents normalization. Interchange fee names are codes used by downstream matching systems. Even changing a space to a dash or expanding an abbreviation breaks the match.

---

### The Column Definition

```
| network | description | number | amount | total | percent | rate |
```

This is the canonical column anchor. We define the exact columns, in exact order. This directly solves the format inconsistency problem. The model now has a concrete schema to fill, not an open-ended table to invent.

---

### The TBA Rule

```
- If a fee line contains "TBA" or is missing a value, write "TBA" in the
  corresponding field.
```

This is important for data pipeline integrity downstream. Without this, a missing value might be returned as blank, null, empty string, "N/A", "not available" — any number of things. The pipeline consuming this output needs a predictable token. TBA is that token.

---

### The Multi-Rate Rule

```
- If a fee line contains multiple rates or fees, list each as a separate row.
```

Some interchange programs have tiered rates — a different rate for debit versus credit, or domestic versus international. Without this instruction, the model would try to combine them into one row, either averaging the rates or picking one. This instruction forces proper normalization.

---

### The Output Purity Rules

```
- Do not include any commentary or explanation—just the table.
- Your response must start directly with the markdown table, no introductory
  text, no explanation, no preamble whatsoever, the very first character of
  your response must be '|'.
```

This solves the format inconsistency problem at the output boundary. LLMs naturally want to say something before delivering a result — "Here is the extracted table:" or "Based on my analysis...". That preamble breaks any downstream parser that expects raw markdown. The instruction that the first character must be `|` is a hard anchor. It completely eliminates any possibility of preamble because the model knows the very first character is constrained.

---

### The Example Output

```
| description                              | Rate (Percent) | Per-Transaction Fee |
|------------------------------------------|----------------|---------------------|
| VS MANUAL CASH DISBURSEMENT              | 0.00           | $2.00               |
| VS ATM CASH DISBURSEMENT                 | 0.00           | $0.50               |
...
```

This is a few-shot example, and it's one of the most powerful techniques in prompt engineering. When you show the model a concrete example of exactly what you want, it anchors the format far more strongly than any written instruction alone. The model has seen millions of table formats in training — this example says "out of all possible table formats, this exact one." The spacing, the alignment, the dollar sign format, the decimal representation — all of it gets anchored here.

---

### The CANONICAL COLUMNS and RULES Footer

```
CANONICAL COLUMNS (EXACT ORDER):
| network | description | number | amount | total | percent | rate |

RULES:
- Extract rate and percent ONLY if explicitly shown in the table.
- Do NOT infer or calculate values that are not present.
- Normalize column meanings (Txn Count, Volume, etc.).
- Leave missing values blank.
- Do NOT output headers as data.
- Return ONLY the tables.
```

We repeat the canonical columns at the bottom. This is deliberate. In a long prompt, the model reads everything, but the beginning and end carry the most weight in terms of what the model prioritizes when generating output. Repeating the column schema at the very end means it's the last thing the model sees before it starts generating — which anchors the output format right at generation time.

The rules at the bottom are a final guard layer. "Do NOT output headers as data" — we saw this happen early on where the model would sometimes include column header names as the first data row. "Return ONLY the tables" — a final purity check to prevent any trailing commentary after the table ends.

---

## Closing

So that's the full prompt. What I want you to take away is that this isn't over-engineering. Every single line traces back to a real failure we observed. The prompt is essentially a written record of everything that went wrong and how we fixed it. That's what production prompt engineering actually looks like — not writing instructions, but progressively closing the gap between what the model wants to do and what you need it to do.
