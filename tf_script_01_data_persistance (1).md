# SCRIPT 01 — `data-persistance` Module
### "Walking the Code: main.tf, outputs.tf, variables.tf"

---

## OPENING

"Let's start with `data-persistance` — this module runs first, and everything else depends on it. I'll walk through the code top to bottom, resource by resource."

---

## `locals` — Prefix

```hcl
locals {
  prefix = "wp-phoenix-statement-reporting"
}
```

"First thing in the file is the local prefix — `wp-phoenix-statement-reporting`. Every resource name in this module starts with this. It keeps all our resources identifiable in the AWS console at a glance and avoids naming conflicts with other teams in the same account."

---

## `aws_s3_bucket` — Document Bucket

```hcl
resource "aws_s3_bucket" "phoenix_statement_reporting_bucket" {
  bucket = "${local.prefix}-document-bucket"
}
```

"First resource is the document bucket — `wp-phoenix-statement-reporting-document-bucket`. This is the main working bucket for the pipeline. When a merchant statement PDF comes in, Textract reads it from here, processes it, and the pipeline writes raw and redacted text outputs back here. This bucket is the central storage the data pipeline reads from and writes to throughout processing."

---

## `data "aws_iam_policy_document"` — SageMaker S3 Access Policy

```hcl
data "aws_iam_policy_document" "sagemaker_s3_access_phoenix_user_access" {
  statement {
    sid    = "SageMakerS3Access"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = [var.wp_phoenix_statment_reporting_sagemaker_execution_role_arn]
    }
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [bucket.arn, "${bucket.arn}/*"]
  }
}
```

"This is a bucket policy document — it defines who can access the document bucket and what they can do. The principal is the SageMaker execution role, which comes in as a variable. It gets `GetObject`, `PutObject`, and `ListBucket` — the minimum needed to read and write documents. The SageMaker execution role ARN is passed in as a variable rather than hardcoded so the data-persistance module doesn't need to know how the compute module names its role."

---

## `aws_s3_bucket_policy` — Attach Policy to Document Bucket

```hcl
resource "aws_s3_bucket_policy" "sagemaker_bucket_policy" {
  bucket = aws_s3_bucket.phoenix_statement_reporting_bucket.id
  policy = data.aws_iam_policy_document.sagemaker_s3_access_phoenix_user_access.json
}
```

"This attaches the policy document we just defined to the document bucket. In Terraform, defining a policy document and attaching it are always two separate steps — the `data` block defines the JSON, the resource block applies it."

---

## `aws_cloudwatch_log_group` — OpenSearch Logs

```hcl
resource "aws_cloudwatch_log_group" "cloudwatch_logs_opensearch" {
  name = "${local.prefix}-opensearch-cloudwatch-logs"
}
```

"Before we define OpenSearch itself, we create the CloudWatch log group it will write to. OpenSearch can publish slow query logs, and having those in CloudWatch means we can monitor and debug search performance from the AWS console. We create the log group first because OpenSearch needs to reference it."

---

## `data "aws_iam_policy_document"` — CloudWatch Access for OpenSearch

```hcl
data "aws_iam_policy_document" "cloudwatch_access_to_opensearch" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["es.amazonaws.com"]
    }
    actions   = ["logs:PutLogEvents", "logs:PutLogEventsBatch", "logs:CreateLogStream"]
    resources = ["arn:aws:logs:*"]
  }
}
```

"This policy document grants the OpenSearch service — `es.amazonaws.com` — permission to write logs into CloudWatch. OpenSearch is the principal here, not a role. This is what allows the log publishing to actually work."

---

## `aws_cloudwatch_log_resource_policy` — Attach CloudWatch Policy

```hcl
resource "aws_cloudwatch_log_resource_policy" "cloudwatch_resource_policy_opensearch" {
  policy_name     = "${local.prefix}-cloudwatch-resource-policy"
  policy_document = data.aws_iam_policy_document.cloudwatch_access_to_opensearch.json
}
```

"Same pattern — we defined the policy document above, now we attach it as a CloudWatch resource policy. This is what actually grants OpenSearch permission to write to the log group."

---

## `data "aws_iam_policy_document"` — OpenSearch Access Policy for SageMaker

```hcl
data "aws_iam_policy_document" "opensearch_access_policy" {
  statement {
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = [var.wp_phoenix_statment_reporting_sagemaker_execution_role_arn]
    }
    actions   = ["es:*"]
    resources = ["arn:aws:es:${var.region}:${var.account_id}:domain/wp-phoenix-opensearch-db/*"]
  }
}
```

"This grants the SageMaker execution role full access to our OpenSearch domain. The `es:*` wildcard is broad — in production you'd scope this down to only the operations the application actually needs, like `ESHttpGet`, `ESHttpPost`, `ESHttpPut`. For the POC, `es:*` keeps things simple. The resource is scoped to our specific domain ARN, not all OpenSearch domains in the account."

---

## `aws_opensearch_domain` — Vector Database

```hcl
resource "aws_opensearch_domain" "opensearch_vector_database" {
  domain_name    = "wp-phoenix-opensearch-db"
  engine_version = "Elasticsearch_7.10"

  cluster_config {
    instance_type = "t3.small.search"  # r6g.large.search for prod
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 10
    volume_type = "gp2"
  }

  access_policies = data.aws_iam_policy_document.opensearch_access_policy.json

  log_publishing_options {
    cloudwatch_log_group_arn = aws_cloudwatch_log_group.cloudwatch_logs_opensearch.arn
    log_type                 = "INDEX_SLOW_LOGS"
  }
}
```

"This is the OpenSearch domain — our vector database that powers the embedding and retrieval pipelines. A few decisions worth calling out:

`engine_version = 'Elasticsearch_7.10'` — we use the Elasticsearch-compatible engine because the k-NN vector search plugin is well-supported on this version.

`instance_type = 't3.small.search'` — smallest available instance, suitable for a POC with limited document volume. The comment in the code already notes `r6g.large.search for prod` — so the production upgrade path is documented right here.

`volume_size = 10` with `gp2` — 10GB is enough for the POC. Production would need sizing based on vector count and document volume.

`log_type = 'INDEX_SLOW_LOGS'` — we publish index slow logs, which helps diagnose performance issues when indexing documents gets slow.

The `access_policies` field wires in the SageMaker access policy we defined just above."

---

## `aws_dynamodb_table` — Single Table

```hcl
resource "aws_dynamodb_table" "phoenix_statement_reporting_table" {
  name         = "${local.prefix}-table"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"

  attribute { name = "PK",     type = "S" }
  attribute { name = "SK",     type = "S" }
  attribute { name = "GSI1PK", type = "S" }
  attribute { name = "GSI1SK", type = "S" }
  attribute { name = "GSI2PK", type = "S" }
  attribute { name = "GSI2SK", type = "S" }

  global_secondary_index {
    name            = "GSI1"
    hash_key        = "GSI1PK"
    range_key       = "GSI1SK"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "GSI2"
    hash_key        = "GSI2PK"
    range_key       = "GSI2SK"
    projection_type = "ALL"
  }
}
```

"This is our DynamoDB table. A few design decisions here:

`billing_mode = 'PAY_PER_REQUEST'` — on-demand billing. For a POC with variable and unpredictable traffic, you don't want to guess at read/write capacity units upfront. Pay-per-request scales automatically and you only pay for what you use.

`hash_key = 'PK'`, `range_key = 'SK'` — composite primary key, both strings. This is single-table design — all entity types in the system live in this one table, differentiated by their PK and SK values. Document records, HITL records, evaluation records — all in here. This is the table the `dynamodb_all.py` functions we covered in the data pipeline and HITL sessions read from and write to.

Two Global Secondary Indexes — `GSI1` and `GSI2` — each with their own PK/SK. GSIs let us query the table by different keys than the primary key. Each GSI projects `ALL` attributes, meaning you get the full item back on a GSI query without needing a second lookup."

---

## `aws_s3_bucket` — Snowflake Bucket

```hcl
resource "aws_s3_bucket" "snowflake_bucket" {
  bucket = "${local.prefix}-snowflake-bucket"
  tags = {
    Purpose = "snowflake"
  }
}
```

"The Snowflake staging bucket. Once documents are processed, the extracted financial data can be staged here and picked up by Snowflake for downstream analytics. The bucket is active. The Snowflake IAM role that would allow Snowflake to assume access to it is commented out — we'll get to that."

---

## `aws_s3_bucket` — Lambda Bucket

```hcl
resource "aws_s3_bucket" "lambda_bucket" {
  bucket = "${local.prefix}-lambda-bucket"
  tags = {
    Purpose = "lambda"
  }
}
```

"The Lambda deployment bucket. When we update a Lambda function — `hitl_wait`, `hitl_resolve`, or the processor Lambda — we zip the code and upload it here first. Then Terraform in the compute module points the Lambda function at the new zip file in this bucket. This is the handoff point between the Python code and the Terraform-managed Lambda resources."

---

## Commented-out Snowflake IAM Role

```hcl
# resource "aws_iam_role" "snowflake_role" {
#   name = "${local.prefix}-snowflake-role"
#   assume_role_policy = jsonencode({ ... })
# }
```

"This is the IAM role that would allow Snowflake to assume access to the Snowflake S3 bucket. It's commented out because completing this integration requires the correct Snowflake external ID and Snowflake AWS account ID, which need to come from the Snowflake admin. The scaffolding is all here — the variables are defined, the role structure is written. When the integration is ready to complete, it's a matter of filling in the real values and uncommenting this block and its policy attachment below."

---

## `aws_iam_policy` — Lambda S3 Policy

```hcl
resource "aws_iam_policy" "lambda-s3-policy" {
  name   = "${local.prefix}-lambda-s3-policy"
  policy = jsonencode({
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = lambda_bucket.arn },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        Resource = "${lambda_bucket.arn}/*" }
    ]
  })
}
```

"IAM policy for the Lambda bucket — `ListBucket` on the bucket itself and `GetObject`, `PutObject`, `DeleteObject` on objects inside it. This pattern of separating bucket-level and object-level permissions is standard AWS practice — `ListBucket` is a bucket-level action, object operations are resource-level. You need both to work with S3 properly."

---

## `aws_iam_policy` — Snowflake S3 Policy

```hcl
resource "aws_iam_policy" "snowflake_s3_policy" { ... }
```

"Same pattern for the Snowflake bucket. This policy is defined and active, ready to be attached to the Snowflake IAM role once that integration is completed."

---

## Commented-out Snowflake Policy Attachment

```hcl
# resource "aws_iam_role_policy_attachment" "snowflake_attach" { ... }
```

"The attachment that would wire the Snowflake S3 policy to the Snowflake role. Also commented out — same reason as the role. When the role gets uncommented, this comes back too."

---

## `aws_s3_bucket` — Code Backup Bucket

```hcl
resource "aws_s3_bucket" "code_backup_bucket" {
  bucket = "${local.prefix}-code-backup"
  tags = {
    Purpose = "code-backup"
  }
}
```

"The last resource in the file. SageMaker notebook instances don't persist their filesystem when stopped — if a developer stops their notebook without pushing their code, it's gone. This bucket is where developers back up their notebook code between sessions. Each developer can push their notebooks here and pull them back when they restart their instance."

---

## `outputs.tf` — What Gets Exported

```hcl
output "phoenix_statement_reporting_bucket_arn" { ... }
output "dynamodb_table_name" { ... }
output "dynamodb_table_arn" { ... }
output "phoenix_statement_reporting_snowflake_bucket_arn" { ... }
output "phoenix_statement_reporting_lambda_bucket_arn" { ... }
output "code_backup_bucket_arn" { ... }
```

"Six outputs — the ARNs and names of the key resources. These are read by the compute workspace via `tfe_outputs`. The compute module uses them in IAM policies and Lambda configurations. The Snowflake role ARN output is commented out — same as the role itself.

You saw this file in VS Code — Image 9 in the walkthrough."

---

## `variables.tf` — Inputs

"The variables file defines the inputs this module accepts. Most have defaults — the region, account ID, environment, and the SageMaker execution role ARN. The SageMaker role ARN is passed in as a variable because the role itself is created in the compute module, and this module needs to reference it in bucket policies. The Snowflake variables — external ID and AWS account ID — are stubbed with placeholder defaults, ready to be filled in when that integration is completed."

---
