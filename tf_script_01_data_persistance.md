# SCRIPT 01 — `data-persistance` Module
### "Storage Layer: S3, DynamoDB, OpenSearch"

---

## OPENING

"Let's start with `data-persistance` — this module runs first, and everything else depends on it. It provisions all the storage resources the system needs: S3 buckets, DynamoDB, and OpenSearch. Once these are up, the compute module can reference them.

The module outputs its resource ARNs and names, and the compute module reads them in via `tfe_outputs`. That's how the two workspaces stay in sync without hardcoding ARNs."

---

## S3 BUCKETS — 4 Buckets, Each with a Purpose

"We provision 4 S3 buckets, each with a specific role:

**Document bucket** — `wp-phoenix-statement-reporting-document-bucket`. This is the main working bucket. When a merchant statement PDF comes in, Textract uploads it here, processes it, and writes the raw and redacted text outputs back here. The data pipeline reads from and writes to this bucket throughout processing.

**Snowflake bucket** — `wp-phoenix-statement-reporting-snowflake-bucket`. This is reserved for Snowflake integration. The idea is that once documents are processed, the extracted financial data can be staged here and picked up by Snowflake for analytics downstream. The bucket exists and the IAM policy for it is defined — you can see a `snowflake_s3_policy` in the code. The IAM role for Snowflake to assume is commented out.

**Lambda bucket** — `wp-phoenix-statement-reporting-lambda-bucket`. This is where we upload our Lambda deployment packages — the `.zip` files for `hitl_wait`, `hitl_resolve`, and the processor Lambda. When Terraform provisions the Lambda functions in the compute module, it pulls the code from this bucket. So the workflow is: zip your Lambda code, upload to this bucket, then Terraform creates or updates the function.

**Code backup bucket** — `wp-phoenix-statement-reporting-code-backup`. This is a backup bucket for SageMaker notebook code. Since SageMaker notebooks don't persist storage by default when an instance is stopped, developers can push their notebook code here to make sure nothing is lost between sessions.

Why 4 separate buckets instead of one? Separation of concerns — different access policies, different purposes, easier to audit. In a production system you'd also add lifecycle policies, versioning, and server-side encryption. For the POC we kept it simple — no encryption, no versioning — but the structure is already set up to support that."

---

## DYNAMODB — Single Table, Two GSIs

"Next is DynamoDB. We have one table — `wp-phoenix-statement-reporting-table` — and it uses a single-table design pattern.

The table has a composite primary key: `PK` as the hash key and `SK` as the range key, both strings. On top of that we have two Global Secondary Indexes — `GSI1` and `GSI2` — each with their own `PK` and `SK` attributes.

The billing mode is `PAY_PER_REQUEST` — also called on-demand. We chose this over provisioned capacity because for a POC with variable and unpredictable traffic, you don't want to be guessing at read/write capacity units. With on-demand you pay per request and DynamoDB scales automatically. The downside is it can get expensive at high scale, but for a POC that's fine.

The single-table design means all entity types — document metadata, HITL records, evaluation records, extraction records — live in the same table, differentiated by their PK and SK patterns. This is more efficient than multiple tables and allows the access patterns we built in `dynamodb_all.py` — the patterns we covered in the data pipeline and HITL sessions."

---

## OPENSEARCH — Vector Database for Embeddings

"OpenSearch is the vector database that powers the embedding and retrieval pipelines.

We're running `Elasticsearch_7.10` engine on a `t3.small.search` instance with a 10GB `gp2` EBS volume. We also have CloudWatch logging enabled — index slow logs go to a dedicated log group so we can monitor and debug search performance.

The access policy grants the SageMaker execution role full `es:*` access to the domain. In production, you'd scope this down to only the operations the application actually needs.

Why `t3.small.search` and not something larger? POC — it's the smallest instance that works. The comment in the code actually calls this out: `# r6g.large.search for prod`. So the production instance type is already noted. For a POC with limited document volume, `t3.small` is sufficient and keeps costs down.

Why OpenSearch over a dedicated vector DB like Pinecone or Weaviate? A few reasons. First, OpenSearch is an AWS-native service — no extra vendor, no extra credentials, works within our existing IAM setup. Second, it supports k-NN vector search natively. Third, for a POC where we're not running millions of vectors, OpenSearch is more than capable. If the product scales significantly, migrating to a purpose-built vector DB would be worth evaluating."

---

## SNOWFLAKE ROLE — COMMENTED OUT

"You'll notice the Snowflake IAM role and its policy attachment are commented out. The bucket and the IAM policy for it exist — those are active. But the IAM role that would allow Snowflake to assume access to the bucket is commented out.

The reason: setting up the Snowflake trust relationship requires coordination with the Snowflake admin to get the correct external ID and Snowflake AWS account ID. We stubbed out the variables and the role structure so it's ready to enable, but we didn't complete that integration in the POC. When the time comes, it's a matter of filling in the real values and uncommenting those blocks."

---

## OUTPUTS — Passed to Compute

"Finally, the outputs. This module exports 6 values: the document bucket ARN, DynamoDB table name, DynamoDB table ARN, Snowflake bucket ARN, Lambda bucket ARN, and code backup bucket ARN.

These are all marked as `nonsensitive` in the compute module's `tfe_outputs` call, which means Terraform Cloud can pass them between workspaces without treating them as secrets. The compute module picks them up and uses them in IAM policies and Lambda configs — so there's no hardcoding of ARNs across modules."

---

## IMPROVEMENTS FOR PRODUCTION

"A few things that would need to change moving from POC to production:

S3 buckets should have versioning enabled, server-side encryption, and bucket policies that block public access explicitly. Right now none of that is configured.

DynamoDB should have point-in-time recovery enabled, and you'd want to think carefully about the GSI access patterns as volume grows.

OpenSearch should move to a larger instance type with multi-AZ replication and encryption at rest.

The Snowflake integration needs to be completed — the scaffolding is there.

All of these are known gaps — they were intentional trade-offs to keep the POC moving fast."

---
