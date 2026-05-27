# POC Infrastructure — Approach Options
### "What to do with the infrastructure while the project is on hold"

---

## CONTEXT

Project is on hold. Two questions to answer:
1. What does it cost to just leave everything running as-is?
2. What are our options — from full teardown to minimal hold?

---

## CURRENT COST — If We Do Nothing (Everything Running)

| Resource | Notes | Est. Monthly Cost |
|---|---|---|
| SageMaker `ml.t3.xlarge` x4 | If all 4 running 24/7 | ~$200–240 |
| NAT Gateway | Per-hour + data processing | ~$35–40 |
| OpenSearch `t3.small.search` | Running 24/7, 10GB gp2 | ~$28–30 |
| DynamoDB on-demand | Near zero with no traffic | ~$0–2 |
| S3 (4 buckets) | Storage only, near zero data | ~$1–3 |
| Lambda / Step Functions | Zero with no invocations | ~$0 |

**Total as-is: ~$265–315/month**

The biggest driver by far is the **4 SageMaker notebooks** (~$220/month). Everything else is secondary.

> **Note: NAT Gateway + Elastic IP are deleted in ALL options below.** They were provisioned but never actually used — no traffic has ever routed through them. No data loss, no impact on anything.

---

## OPTION 1 — Full Destroy ($0/month)

**What:** Run `terraform destroy` on all 3 workspaces. Everything gone.

**Cost:** $0/month

**Changes:**
- Delete NAT Gateway + Elastic IP
- Stop and delete all 4 SageMaker notebooks
- Delete OpenSearch domain (delete data first)
- Destroy all remaining resources: DynamoDB, S3, Lambdas, Step Functions, IAM roles

**To get back for a demo (~1 hour):**
1. Run `terraform apply` on `data-persistance` (~10–15 min to provision OpenSearch)
2. Run `terraform apply` on `compute` (~5 min for notebooks, Lambdas, Step Functions)
3. Re-upload Lambda zip files to S3
4. Re-push embeddings to OpenSearch from documents (~15–20 min)

**Pros:**
- Zero cost
- Clean slate
- Nothing is truly lost — all code and Terraform state stays in GitHub and Terraform Cloud

**Cons:**
- ~1 hour of setup needed before any demo
- Need to re-upload Lambda zips and re-index embeddings each time

**Destroy order matters:**
1. Destroy `compute` first (depends on `data-persistance` outputs)
2. Then `data-persistance`
3. Then `core-networking`

---

## OPTION 2 — Keep OpenSearch, Stop Notebooks ✅ RECOMMENDED

**What:** Delete NAT + EIP, stop all notebooks, keep OpenSearch and everything else running.

**Cost: ~$28–33/month**

**Changes:**
- Delete NAT Gateway + Elastic IP → saves ~$35–40/month
- Stop all 4 SageMaker notebooks → $0 while stopped
- Keep OpenSearch running → embeddings stay intact, no re-indexing needed
- Keep everything else: DynamoDB, S3, Lambdas, Step Functions, IAM roles

**To get back for a demo (~5 minutes):**
- Start 1 SageMaker notebook (~2 min in console)
- Everything else is already live — embeddings in OpenSearch, DynamoDB intact, Lambdas ready
- That's it ✓

**Pros:**
- Fastest path to demo — no Terraform apply, no re-indexing
- All embeddings and data preserved exactly as-is
- Only ~$28–33/month ongoing

**Cons:**
- ~$28–33/month for OpenSearch sitting idle
- If project never comes back, that's wasted spend

**Why recommended:** The $28/month buys instant demo readiness. Given a stakeholder could ask for a demo with short notice, this is the best balance of cost and responsiveness.

---

## OPTION 3 — Minimal Hold (~$1–5/month)

**What:** Delete NAT + EIP, stop notebooks, delete OpenSearch. Keep only the truly cheap resources.

**Cost: ~$1–5/month** (just S3 storage)

**Changes:**
- Delete NAT Gateway + Elastic IP
- Stop all 4 SageMaker notebooks
- Delete OpenSearch domain (delete data first)
- Keep: DynamoDB, S3 buckets, Lambdas, Step Functions, IAM roles

**Remaining resources (all near-zero cost):**
- DynamoDB — pay per request, $0 with no traffic
- 4 S3 buckets — ~$1–3/month total storage
- 3 Lambda functions — $0 with no invocations
- Step Functions state machine — $0 with no executions
- All IAM roles and policies — always free

**To get back for a demo (~1 hour):**
1. Run `terraform apply` on `data-persistance` to recreate OpenSearch (~10–15 min)
2. Start 1 SageMaker notebook (~2 min)
3. Re-push embeddings from documents (~15–20 min)

**Pros:**
- Near-zero cost
- All meaningful scaffolding (IAM, Lambda, DynamoDB, S3) intact — not starting from scratch
- Faster to restore than full destroy

**Cons:**
- ~1 hour to demo-ready vs ~5 min for Option 2
- Need to re-push embeddings each time

---

## SUMMARY COMPARISON

| | Option 1 | Option 2 ✅ | Option 3 |
|---|---|---|---|
| **Monthly cost** | $0 | ~$28–33 | ~$1–5 |
| **Demo ready in** | ~1 hour | ~5 minutes | ~1 hour |
| **OpenSearch** | Deleted | ✅ Running | Deleted |
| **Notebooks** | Deleted | Stopped | Stopped |
| **DynamoDB / S3 / Lambda** | Deleted | ✅ Intact | ✅ Intact |
| **NAT Gateway** | Deleted | Deleted | Deleted |

---

## WHAT TO TELL YOUR TECH LEAD

"NAT Gateway and Elastic IP get deleted in all cases — they were never actually used. For everything else we have three options. Option 1 is full destroy at $0 — clean slate, about an hour to get back. Option 2 keeps OpenSearch running and stops the notebooks at ~$28–33/month — demo-ready in 5 minutes if a stakeholder asks. Option 3 deletes OpenSearch too and drops cost to ~$1–5/month, but also needs about an hour to restore. We recommend Option 2 — the $28/month is worth the instant demo readiness given the project could be picked up at any time."

---

> ⚠️ **Cost Disclaimer:** All figures above are rough estimates based on publicly available AWS pricing and are intended for high-level planning only. Actual costs will vary based on your specific account, region, any reserved instance pricing, data transfer, and other factors. We recommend checking the AWS Pricing Calculator for accurate figures before making a final decision.

---
