# SCRIPT 02 — `compute` Module
### "Compute Layer: SageMaker, Lambda, Step Functions, IAM, Bedrock"

---

## OPENING

"Now let's go through the `compute` module — this is the biggest one, 28 resources. It reads all its storage references from the `data-persistance` workspace outputs, and then provisions everything that actually runs the system: SageMaker notebooks, Lambda functions, Step Functions, all the IAM roles and policies, and the Bedrock guardrail.

I'll go section by section."

---

## CROSS-WORKSPACE DEPENDENCY — `tfe_outputs`

"The very first thing in `compute/main.tf` is this:

```hcl
data "tfe_outputs" "phoenix_data_persist" {
  organization = "worldpay-tf"
  workspace    = "phoenix-statement-reporting-data-persistance-dev"
}
```

This is how the compute workspace reads outputs from the data-persistance workspace. Terraform Cloud makes these available across workspaces. We then pull the specific values into locals — DynamoDB ARN, all the S3 bucket ARNs — and reference those locals throughout the rest of the module. This means if the DynamoDB table ever gets recreated with a new ARN, all the IAM policies in compute automatically pick up the new value on the next run. No manual updates needed."

---

## BEDROCK GUARDRAIL — PII Protection at the LLM Layer

"The first resource is the Bedrock guardrail:

```hcl
resource "aws_bedrock_guardrail" "merchant_statement_pii" {
  name = "merchant-statement-pii-guardrail"
  ...
}
```

This guardrail is actively used in the pipeline. It intercepts inputs and outputs to Bedrock models and anonymizes 5 PII entity types: NAME, PHONE, EMAIL, ADDRESS, and URL. Both input and output anonymization are enabled for each.

Why a Bedrock guardrail on top of Comprehend PII redaction? Defence in depth. Comprehend handles PII in the raw OCR'd document text before it goes into the pipeline. The Bedrock guardrail is a second layer that catches anything that might slip through when the LLM is doing extraction or evaluation. In a financial product handling merchant data, you want multiple layers.

We also version the guardrail — `aws_bedrock_guardrail_version` creates a published `v1`. This is important because the Python code references a specific guardrail version, not just the latest. That way a guardrail change doesn't silently affect running pipelines until you explicitly update the version reference.

For the POC, we have 5 PII types. In production, you'd review the full list of Bedrock-supported PII types and add any that are relevant — things like SSN, credit card numbers, tax IDs, which all appear in merchant statements."

---

## IAM — ROLES AND POLICIES

"IAM is the bulk of the compute module. We have 3 IAM roles: SageMaker execution role, Lambda execution role, and Step Functions role. Let me walk through each."

### SageMaker Execution Role

"The SageMaker execution role is what the notebooks run as. It gets a large set of permissions because the notebooks are where the entire pipeline runs during development.

The permissions cover:
- **Bedrock** — `InvokeModel`, `InvokeModelWithResponseStream`, `ApplyGuardrail` for the specific guardrail we just defined, plus read access to list and describe guardrails
- **DynamoDB** — full CRUD on our specific table and its indexes. Scoped to just our table ARN — not `resources: *`
- **Textract** — all the document analysis operations: `StartDocumentAnalysis`, `GetDocumentAnalysis`, `DetectDocumentText`, `AnalyzeDocument`, `AnalyzeExpense`
- **Comprehend** — PII detection, entity detection, sentiment, key phrases
- **S3** — scoped access to 4 buckets: document bucket, Snowflake bucket, Lambda bucket, and code backup bucket. Each bucket has separate ListBucket and object-level permissions
- **OpenSearch** — `es:*` on our specific domain
- **EC2 read** — describe subnets, network interfaces, route tables, security groups, VPC endpoints. Read-only, needed for network-aware services to understand the environment
- **CloudWatch Logs** — read access to describe and get log events. Useful for debugging from within the notebook

Why scope permissions this tightly? Even for a POC in a dev environment, it's good practice to only grant what's actually needed. It also means when we move to production, we already have a well-scoped role that needs minimal changes."

### Lambda Execution Role

"The Lambda execution role is shared by all 3 Lambda functions. Its permissions are much narrower than SageMaker:
- `AWSLambdaBasicExecutionRole` — the AWS managed policy that lets Lambda write logs to CloudWatch
- `states:SendTaskSuccess` and `states:SendTaskFailure` — this is the critical one. The HITL resolve Lambda needs this to send the callback to Step Functions and resume the paused execution
- S3 get/put on the Lambda bucket — so Lambdas can read their own deployment packages if needed
- DynamoDB `PutItem`, `UpdateItem`, `GetItem` — the minimum needed for the HITL wait Lambda to store the task token

Why not give Lambda the same broad permissions as SageMaker? Principle of least privilege. The Lambdas have one job each — store a token, or send a callback. They don't need Textract, Comprehend, or Bedrock access."

### Step Functions Role

"The Step Functions role is the simplest. It only needs one permission: `lambda:InvokeFunction` on the `hitl_wait` Lambda. That's it. Step Functions invokes the Lambda with the task token, then pauses. The role doesn't need DynamoDB, S3, or anything else."

---

## LAMBDA FUNCTIONS — 3 Functions

"We have 3 Lambda functions, all Python 3.10, all pulling their deployment packages from the Lambda S3 bucket.

**`phoenix_lambda`** — This is the main FastAPI application packaged as a Lambda. Handler is `lambda_function.lambda_handler`. 512MB memory, 15-second timeout. This is the processor that runs the full pipeline — data, embedding, retrieval, evaluation. The timeout of 15 seconds is tight for a full pipeline run; in production you'd want to increase this or move to an async pattern.

**`hitl_wait`** — Handler is `hitl_wait_stub.handler`. 128MB memory, 60-second timeout. This Lambda's only job is to receive the task token from Step Functions and write it to DynamoDB. It's intentionally tiny — 128MB is the minimum, because all it does is one DynamoDB write.

**`hitl_resolve`** — Handler is `hitl_resolve_handler.handler`. 256MB memory, 60-second timeout. This one checks whether all flagged fields have been resolved and calls `SendTaskSuccess` to wake the Step Function back up. A bit more logic than the wait stub, so slightly more memory.

All 3 share the same Lambda execution role — in production you'd want separate roles per function with only the permissions each one needs."

---

## STEP FUNCTIONS — The HITL State Machine

"The Step Functions state machine is defined inline as a JSON-encoded local:

```hcl
WaitForHuman = {
  Type     = "Task"
  Resource = "arn:aws:states:::lambda:invoke.waitForTaskToken"
  ...
}
```

One state, `WaitForHuman`, using the `waitForTaskToken` integration pattern. This is the entire state machine definition. When Step Functions enters this state, it invokes `hitl_wait`, passes the task token in the payload, and then pauses indefinitely. It won't proceed until something calls `SendTaskSuccess` with that token.

We use `STANDARD` type, not `EXPRESS`. Standard executions can run for up to a year and have exactly-once execution semantics. Express is cheaper but only runs for 5 minutes and has at-least-once semantics — not appropriate for a human review workflow that might take hours or days.

The Step Functions role is also provisioned here — it can invoke both `phoenix_lambda` and `hitl_wait`. This is intentional: we may extend the state machine in the future to invoke additional Lambdas."

---

## SAGEMAKER NOTEBOOKS — 4 Instances

"We provision 4 SageMaker notebook instances, all `ml.t3.xlarge`, all using the same SageMaker execution role.

**`notebook`** — Harsha's primary notebook. This is the main development environment for the pipeline. 20GB storage.

**`notebook-dev-2`** — Louden's notebook. Each team member gets their own isolated environment so there's no interference between developers.

**`notebook-dev-3`** — Harsha's second notebook, with 30GB storage. Used specifically for pipeline debugging, testing Terraform resource connections, and as a practice/sandbox environment. The extra storage reflects the additional data it might hold during testing.

**`notebook-dev-4`** — Srijha's notebook. 20GB storage.

Why SageMaker notebooks instead of EC2 instances or local Docker containers? A few reasons. SageMaker notebooks come pre-configured with Python, Jupyter, and AWS SDKs. They run inside the AWS environment, so they have direct network access to our resources — DynamoDB, OpenSearch, S3 — without needing to tunnel or configure credentials locally. They're also easy to stop and start, and cost nothing when stopped. The one caveat is that the filesystem doesn't persist after the instance stops, which is why we have the code backup S3 bucket — developers push their notebooks there to save their work."

---

## OUTPUTS

"The compute module outputs 4 values: the SageMaker execution role ARN, the Bedrock guardrail ID, the full guardrail ARN, the published guardrail version, and the Step Functions state machine ARN.

The guardrail outputs are important because the Python pipeline code needs the guardrail ID and version to call `bedrock:ApplyGuardrail` correctly. Exporting them as Terraform outputs means the application config can reference them without anyone needing to look them up manually in the console."

---

## IMPROVEMENTS FOR PRODUCTION

"A few key improvements for production:

Each Lambda function should have its own IAM role with only the permissions it needs — right now all 3 share one role.

The Lambda timeout for `phoenix_lambda` is 15 seconds, which is tight for a full pipeline run. In production either increase the timeout or move to an async invocation pattern.

SageMaker notebook instances should be in a private subnet with VPC endpoints for AWS services rather than going over the public internet. Currently there's no VPC config on the notebooks.

The Bedrock guardrail should be expanded to cover additional PII types relevant to financial documents.

Step Functions logging should be enabled in production — right now there's no CloudWatch log group attached to the state machine, so debugging failed executions requires digging through CloudWatch manually."

---
