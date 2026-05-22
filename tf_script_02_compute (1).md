# SCRIPT 02 — `compute` Module
### "Walking the Code: main.tf, outputs.tf, variables.tf"

---

## OPENING

"Now `compute` — the biggest module, 28 resources. I'll walk through `main.tf` top to bottom, resource by resource."

---

## `locals` — Prefix

```hcl
locals {
  prefix = "wp-phoenix-statement-reporting"
}
```

"Same prefix as every other module. All resource names start with `wp-phoenix-statement-reporting`."

---

## `data "tfe_outputs"` — Reading from data-persistance

```hcl
data "tfe_outputs" "phoenix_data_persist" {
  organization = "worldpay-tf"
  workspace    = "phoenix-statement-reporting-data-persistance-dev"
}
```

"Right at the top — before any resources — we read the outputs from the data-persistance workspace. This is the cross-workspace dependency. Terraform Cloud makes the outputs of one workspace available to another through this `tfe_outputs` data source."

---

## `locals` — ARNs from data-persistance

```hcl
locals {
  dynamodb_table_arn     = data.tfe_outputs.phoenix_data_persist.nonsensitive_values.dynamodb_table_arn
  dynamodb_table_name    = data.tfe_outputs.phoenix_data_persist.nonsensitive_values.dynamodb_table_name
  documents_bucket_arn   = ...
  snowflake_bucket_arn   = ...
  lambda_bucket_arn      = ...
  code_backup_bucket_arn = ...
}
```

"We immediately pull those outputs into local values — DynamoDB ARN, DynamoDB name, and all 4 S3 bucket ARNs. From here on, every IAM policy and every Lambda config in this module references these locals. If the data-persistance workspace ever recreates a resource with a new ARN, the compute workspace picks it up automatically on the next run. No hardcoded ARNs scattered through the code."

---

## `locals` — Step Function Definition

```hcl
locals {
  step_function_definition = jsonencode({
    Comment = "Phoenix HITL Wait-Only State Machine"
    StartAt = "WaitForHuman"
    States = {
      WaitForHuman = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke.waitForTaskToken"
        Parameters = {
          FunctionName = aws_lambda_function.hitl_wait.arn
          Payload = {
            "task_token.$" = "$$.Task.Token"
            "doc_id.$"     = "$.doc_id"
          }
        }
        End = true
      }
    }
  })
}
```

"The Step Functions state machine definition is defined here as a local, then referenced later when creating the state machine resource. One state — `WaitForHuman` — using the `waitForTaskToken` integration pattern. When Step Functions enters this state, it invokes `hitl_wait`, passes the task token and doc ID in the payload, and then pauses. It will not proceed until something calls `SendTaskSuccess` with that token. This is the entire HITL pause mechanism we covered in the HITL session, defined here in Terraform."

---

## `aws_bedrock_guardrail` — PII Guardrail

```hcl
resource "aws_bedrock_guardrail" "merchant_statement_pii" {
  name        = "merchant-statement-pii-guardrail"
  description = "PII guardrail for merchant statements"

  sensitive_information_policy_config {
    pii_entities_config { type = "NAME",    action = "ANONYMIZE", ... }
    pii_entities_config { type = "PHONE",   action = "ANONYMIZE", ... }
    pii_entities_config { type = "EMAIL",   action = "ANONYMIZE", ... }
    pii_entities_config { type = "ADDRESS", action = "ANONYMIZE", ... }
    pii_entities_config { type = "URL",     action = "ANONYMIZE", ... }
  }

  blocked_input_messaging   = "Input contains sensitive information..."
  blocked_outputs_messaging = "Output contains sensitive information..."
}
```

"The Bedrock guardrail — actively used in the pipeline. It intercepts inputs and outputs to Bedrock models and anonymizes 5 PII types: NAME, PHONE, EMAIL, ADDRESS, URL. Both `input_action` and `output_action` are set to `ANONYMIZE` for each — so PII gets masked on the way in and on the way out.

Why this on top of Comprehend's PII redaction? Defence in depth. Comprehend handles raw OCR text before it enters the pipeline. The Bedrock guardrail is a second layer at the LLM boundary. In a financial product handling merchant data, two layers of PII protection is the right call.

The blocked messaging fields define what gets returned to the caller if PII is detected and blocked — visible here in the Terraform, consistent across all environments."

---

## `aws_bedrock_guardrail_version` — Published Version

```hcl
resource "aws_bedrock_guardrail_version" "merchant_statement_pii_v1" {
  guardrail_arn = aws_bedrock_guardrail.merchant_statement_pii.guardrail_arn
  description   = "v1 - initial published PII guardrail"
}
```

"We publish an explicit version of the guardrail. The Python pipeline code references a specific version, not `DRAFT`. This matters — if someone updates the guardrail config, the running pipeline won't be affected until you explicitly publish a new version and update the reference in the code. Gives us control over when guardrail changes take effect."

---

## `data "aws_iam_policy_document"` — SageMaker Permissions

```hcl
data "aws_iam_policy_document" "sagemaker_permissions" {
  statement { sid = "BedrockAccess" ... }
  statement { sid = "BedrockGuardrailRead" ... }
  statement { sid = "DynamoDBAccessForNotebook" ... }
  statement { sid = "AllowApplySpecificGuardrail" ... }
  statement { sid = "TextractAccess" ... }
  statement { sid = "ComprehendAccess" ... }
  statement { sid = "BedrockTitanTextInvoke" ... }
  statement { sid = "AllowReadCloudWatchLogs" ... }
  statement { sid = "S3Access" ... }
  statement { sid = "S3ObjectAccess" ... }
  statement { sid = "CodeBackupListBucket" ... }
  statement { sid = "CodeBackupObjectAccess" ... }
  statement { sid = "SnowflakeListBucket" ... }
  statement { sid = "SnowflakeObjectAccess" ... }
  statement { sid = "lambdaListBucket" ... }
  statement { sid = "lambdaObjectAccess" ... }
  statement { sid = "EC2Read" ... }
  statement { sid = "OpenSearchAccess" ... }
}
```

"This is the big IAM policy document for the SageMaker execution role — the role the notebooks run as. It's large because the notebooks run the entire pipeline during development, so the role needs access to every service the pipeline touches.

Let me go through the statement groups:

**Bedrock** — `InvokeModel` and `InvokeModelWithResponseStream` for calling LLM models. Separate statement for `BedrockGuardrailRead` — list and describe guardrails. And `AllowApplySpecificGuardrail` scoped to the specific guardrail ARN we just created. Not `resources: *` — just our guardrail.

**DynamoDB** — Full CRUD on our specific table and its indexes. Scoped to `local.dynamodb_table_arn` — not all DynamoDB tables in the account.

**Textract** — All document analysis operations: `StartDocumentAnalysis`, `GetDocumentAnalysis`, `DetectDocumentText`, `AnalyzeDocument`, `AnalyzeExpense`. These are what `textract.py` calls.

**Comprehend** — PII detection, entity detection, sentiment, key phrases. These are what `comprehend.py` calls.

**Bedrock Titan** — Specific permission to invoke the Titan embedding model. Scoped to the exact model ARN.

**CloudWatch Logs** — Read-only access to describe and get log events. Useful for debugging from inside the notebook.

**S3** — Four separate bucket access blocks, each with `ListBucket` on the bucket and object-level permissions on its contents. Document bucket, code backup, Snowflake, Lambda. Each bucket scoped individually — no wildcard S3 access.

**EC2 Read** — Describe subnets, network interfaces, route tables, security groups, VPC endpoints. Read-only. Needed for network-aware services to understand the VPC topology.

**OpenSearch** — `es:*` on our specific domain. Full access but scoped to our domain ARN."

---

## `data "aws_iam_policy_document"` — SageMaker Trust Policy

```hcl
data "aws_iam_policy_document" "sagemaker_trust_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}
```

"The trust policy for the SageMaker role — allows the SageMaker service to assume this role. This is how IAM roles work: the permissions policy says what the role can do, the trust policy says who can assume it."

---

## `data "aws_iam_policy_document"` — Lambda Trust Policy

```hcl
data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}
```

"Same pattern — trust policy for the Lambda execution role. Allows the Lambda service to assume it."

---

## `aws_iam_role` — Lambda Execution Role

```hcl
resource "aws_iam_role" "lambda_execution_role" {
  name               = "${local.prefix}-lambda-execution-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}
```

"The Lambda execution role — shared by all 3 Lambda functions. Created here with the trust policy we just defined."

---

## `aws_iam_role_policy_attachment` — Lambda Basic Logs

```hcl
resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
```

"We attach the AWS managed `AWSLambdaBasicExecutionRole` policy. This is the minimum policy a Lambda needs — it allows the function to write logs to CloudWatch. Without this, Lambda executions are completely silent."

---

## `data "aws_iam_policy_document"` — Lambda Access Policy

```hcl
data "aws_iam_policy_document" "lambda_access" {
  statement {
    sid     = "AllowStepFunctionCallback"
    actions = ["states:SendTaskSuccess", "states:SendTaskFailure"]
    resources = ["*"]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${local.lambda_bucket_arn}/*"]
  }
  statement {
    sid     = "AllowDynamoDBWriteForHITL"
    actions = ["dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:GetItem"]
    resources = [local.dynamodb_table_arn, "${local.dynamodb_table_arn}/index/*"]
  }
}
```

"The Lambda-specific permissions — deliberately much narrower than SageMaker:

`SendTaskSuccess` and `SendTaskFailure` — the HITL resolve Lambda needs these to send the callback to Step Functions and resume the paused execution. This is the key permission for the whole HITL flow.

S3 `GetObject` and `PutObject` on the Lambda bucket — so Lambdas can access their deployment packages if needed.

DynamoDB `PutItem`, `UpdateItem`, `GetItem` — the minimum needed for `hitl_wait` to store the task token and `hitl_resolve` to read and update it.

No Textract, no Comprehend, no Bedrock — Lambdas don't need those. Principle of least privilege."

---

## `aws_lambda_function` — processor Lambda

```hcl
resource "aws_lambda_function" "phoenix_lambda" {
  function_name = "${local.prefix}-processor-lambda"
  role          = aws_iam_role.lambda_execution_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.10"
  s3_bucket     = "wp-phoenix-statement-reporting-lambda-bucket"
  s3_key        = "lambda/processor.zip"
  timeout       = 15
  memory_size   = 512
}
```

"The main processor Lambda — this is the FastAPI application packaged as a Lambda. Handler is `lambda_function.lambda_handler`. 512MB memory, 15 second timeout. The code is pulled from the Lambda S3 bucket at `lambda/processor.zip`. This Lambda runs the full pipeline — data, embedding, retrieval, evaluation — triggered from the API layer."

---

## `aws_lambda_function` — hitl_wait

```hcl
resource "aws_lambda_function" "hitl_wait" {
  function_name = "${local.prefix}-hitl-wait"
  handler       = "hitl_wait_stub.handler"
  runtime       = "python3.10"
  s3_key        = "lambda/hitl_wait_v7.zip"
  timeout       = 60
  memory_size   = 128
}
```

"The HITL wait Lambda — `hitl_wait_stub.handler`. 128MB memory, 60 second timeout. The smallest memory allocation because its only job is one DynamoDB write — store the task token Step Functions passes in. The `v7` in the S3 key reflects the iteration — this has been updated 7 times as we refined the HITL flow."

---

## `aws_lambda_function` — hitl_resolve

```hcl
resource "aws_lambda_function" "hitl_resolve" {
  function_name = "${local.prefix}-hitl-resolve"
  handler       = "hitl_resolve_handler.handler"
  runtime       = "python3.10"
  s3_key        = "lambda/hitl_resolve.zip"
  timeout       = 60
  memory_size   = 256
}
```

"The HITL resolve Lambda — `hitl_resolve_handler.handler`. 256MB, more than `hitl_wait` because it has more logic: read the DynamoDB record, check all flagged fields are resolved, then call `SendTaskSuccess` to wake the Step Function. 60 second timeout gives the human reviewer workflow room to breathe."

---

## `data "aws_iam_policy_document"` — SageMaker Invoke Lambda

```hcl
data "aws_iam_policy_document" "sagemaker_invoke_lambda" {
  statement {
    sid       = "SageMakerInvokeLambda"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.phoenix_lambda.arn]
  }
}
```

"Allows SageMaker to invoke the processor Lambda. Scoped to just that one Lambda ARN."

---

## `aws_iam_role_policy` — Attach SageMaker Invoke Lambda

```hcl
resource "aws_iam_role_policy" "sagemaker_invoke_lambda_policy" {
  name   = "${local.prefix}-sagemaker-invoke-lambda"
  role   = aws_iam_role.sagemaker_execution_role.id
  policy = data.aws_iam_policy_document.sagemaker_invoke_lambda.json
}
```

"Attaches the Lambda invoke policy to the SageMaker role. Note this references `sagemaker_execution_role` — that role is defined a few lines further down in the file. Terraform resolves these forward references at plan time."

---

## `aws_iam_role_policy` — Attach Lambda Access Policy

```hcl
resource "aws_iam_role_policy" "lambda_access_policy" {
  role   = aws_iam_role.lambda_execution_role.id
  policy = data.aws_iam_policy_document.lambda_access.json
}
```

"Attaches the Lambda access policy — the `SendTaskSuccess`, S3, and DynamoDB permissions — to the Lambda execution role."

---

## `aws_iam_role` — SageMaker Execution Role

```hcl
resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${local.prefix}-sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust_policy.json
}
```

"The SageMaker execution role — what all 4 notebooks run as. Created with the SageMaker trust policy."

---

## `aws_iam_role_policy` — Attach SageMaker Permissions

```hcl
resource "aws_iam_role_policy" "sagemaker_service_permissions_policy" {
  name   = "${local.prefix}-sagemaker_policy"
  role   = aws_iam_role.sagemaker_execution_role.id
  policy = data.aws_iam_policy_document.sagemaker_permissions.json
}
```

"Attaches the big SageMaker permissions policy — all the Bedrock, Textract, Comprehend, DynamoDB, S3, OpenSearch permissions — to the SageMaker role."

---

## `data "aws_iam_policy_document"` — SageMaker Step Functions Access

```hcl
data "aws_iam_policy_document" "sagemaker_stepfunctions_access" {
  statement { sid = "StartStepFunction" ... actions = ["states:StartExecution"] ... }
  statement { sid = "DescribeAndManageExecutions" ... }
  statement { sid = "AllowSendTaskSuccessForHITLDev" ... }
  statement { sid = "ReadStateMachines" ... }
}
```

"SageMaker needs Step Functions permissions too — the notebooks trigger and monitor the HITL state machine. Four statements:

`StartExecution` — scoped to our specific state machine ARN. SageMaker can only start our state machine, nothing else.

`DescribeAndManageExecutions` — allows the notebook to check execution status, view history, stop an execution if needed.

`AllowSendTaskSuccessForHITLDev` — `SendTaskSuccess` and `SendTaskFailure` scoped to our state machine. This allows the Streamlit reviewer UI running in the notebook to directly resolve a HITL task.

`ReadStateMachines` — describe and list, `resources: *`. Read-only, so safe to be broad."

---

## `aws_iam_role_policy` — Attach SageMaker Step Functions Policy

```hcl
resource "aws_iam_role_policy" "sagemaker_stepfunctions_policy" { ... }
```

"Attaches the Step Functions policy to the SageMaker role."

---

## `data "aws_iam_policy_document"` — Step Functions Trust Policy

```hcl
data "aws_iam_policy_document" "step_functions_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}
```

"Trust policy for the Step Functions role — allows the Step Functions service to assume it."

---

## `aws_iam_role` — Step Functions Role

```hcl
resource "aws_iam_role" "step_functions_role" {
  name               = "${local.prefix}-step-functions-role"
  assume_role_policy = data.aws_iam_policy_document.step_functions_trust.json
}
```

"The Step Functions execution role."

---

## `data "aws_iam_policy_document"` — Step Functions Invoke Lambda

```hcl
data "aws_iam_policy_document" "step_functions_invoke_lambda" {
  statement {
    actions   = ["lambda:InvokeFunction"]
    resources = [
      aws_lambda_function.phoenix_lambda.arn,
      aws_lambda_function.hitl_wait.arn
    ]
  }
}
```

"Step Functions can invoke two Lambdas — the processor and `hitl_wait`. Scoped to just those two ARNs. The state machine currently only invokes `hitl_wait`, but `phoenix_lambda` is included in case we extend the state machine with additional steps in future."

---

## `aws_iam_role_policy` — Attach Step Functions Invoke Lambda

```hcl
resource "aws_iam_role_policy" "step_functions_invoke_lambda_policy" { ... }
```

"Attaches the Lambda invoke policy to the Step Functions role."

---

## `aws_sfn_state_machine` — The State Machine

```hcl
resource "aws_sfn_state_machine" "phoenix_step_function" {
  name       = "${local.prefix}-state-machine"
  role_arn   = aws_iam_role.step_functions_role.arn
  definition = local.step_function_definition
  type       = "STANDARD"
}
```

"The Step Functions state machine. Uses the definition we defined in the local at the top of the file, and the Step Functions role we just created.

`type = 'STANDARD'` — not `EXPRESS`. Standard executions can run for up to a year and have exactly-once execution semantics. EXPRESS is cheaper but only runs for 5 minutes and has at-least-once semantics. A human review workflow that might sit for hours or days must be STANDARD."

---

## `aws_sagemaker_notebook_instance` — 4 Notebooks

```hcl
resource "aws_sagemaker_notebook_instance" "phoenix-statement-reporting-notebook" {
  name          = "${local.prefix}-notebook"
  role_arn      = aws_iam_role.sagemaker_execution_role.arn
  instance_type = "ml.t3.xlarge"
  volume_size   = 20
}
# ...notebook-dev-2, notebook-dev-3, notebook-dev-4
```

"Four SageMaker notebook instances, all `ml.t3.xlarge`, all using the same execution role.

`notebook` — Harsha's primary notebook. 20GB. Main development environment for the pipeline.

`notebook-dev-2` — Louden's notebook. 20GB. Each team member gets their own isolated environment — no interference between developers working on different parts of the pipeline.

`notebook-dev-3` — Harsha's second notebook. 30GB — extra storage because this is used specifically for pipeline debugging, testing Terraform resource connections, and as a sandbox environment. The extra 10GB reflects the additional data it might hold during testing sessions.

`notebook-dev-4` — Srijha's notebook. 20GB.

Why SageMaker notebooks over local Docker or EC2? They come pre-configured with Python and AWS SDKs, they run inside the AWS environment so they have direct network access to all our resources, and they cost nothing when stopped. The code backup S3 bucket we created in data-persistance is there specifically because notebook storage doesn't persist between stops."

---

## `outputs.tf`

```hcl
output "sagemaker_execution_role_arn" { ... }
output "bedrock_guardrail_id" { ... }
output "bedrock_guardrail_arn" { ... }
output "bedrock_guardrail_published_version" { ... }
output "phoenix_step_function_arn" { ... }
```

"Five outputs from the compute module. The guardrail ID, ARN, and published version are exported because the Python pipeline code needs these values to call `bedrock:ApplyGuardrail` correctly. The Step Functions ARN is exported so other parts of the system can reference it. The SageMaker role ARN is exported for reference.

Exporting these as Terraform outputs means no one needs to look them up manually in the console — they're always available in Terraform Cloud."

---

## `variables.tf`

"Variables are minimal — region, account ID, environment, and the source code URL pointing to the GitHub repo. The account ID and region have defaults set to the dev account and `us-east-2`. For a prod deployment these would be overridden with production values."

---
