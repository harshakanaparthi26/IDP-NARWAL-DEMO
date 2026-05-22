# SCRIPT 00 — Terraform KT: Overview, Repo & Workflow
### "How We Manage Infrastructure for Phoenix Statement Reporting"

---

## OPENING

"Alright, so in the previous sessions we walked through the data pipeline and the HITL pipeline — the Python code that runs the system. In this session we're covering the infrastructure layer — Terraform. I'm the only one on the team who owns the Terraform, so I'll be covering everything here, even parts that connect to pipelines owned by other team members.

The goal of this session is to answer three things: what infrastructure we provisioned, why we made those choices, and where we can improve as we move toward production."

---

## THE REPO — `wp-phoenix-ai-infra`

*(point at Image 1 — GitHub repo page)*

"This is our infrastructure repo — `wp-efos/wp-phoenix-ai-infra`. It's private, lives in GitHub Enterprise under the `wp-efos` org. You can see it has 12 branches, 246 commits, and 3 contributors — myself and two other team members.

The repo has two top-level things worth noting. First, `.github/workflows` — that's where our GitHub Actions CI checks live. Second, the `dev/` folder — that's where all the Terraform code lives. Everything is namespaced under `dev` because this is a dev environment. If we were to add staging or prod, they'd sit as sibling folders at the same level."

---

## FOLDER STRUCTURE — 4 Modules

*(point at Image 10 — VS Code file explorer)*

"Inside `dev/` there are 4 modules:

**`ai/`** — This one handles some early AI configuration. We won't focus on it much here.

**`compute/`** — This is the biggest module. It provisions all the compute resources: SageMaker notebooks, Lambda functions, Step Functions, IAM roles, and the Bedrock guardrail. This is the heart of the system.

**`core-networking/`** — This handles networking — NAT gateway, Elastic IP. We'll talk about what we deployed here and why we paused on some parts.

**`data-persistance/`** — This provisions all storage: S3 buckets, DynamoDB, OpenSearch. It runs first because the compute module reads its outputs.

Each module follows the same file pattern: `main.tf`, `outputs.tf`, `variables.tf`, `providers.tf`, and `terraform.tf`. This consistency makes it easy for anyone on the team to navigate any module without needing a guide."

---

## THE WORKFLOW — Code to Cloud

"Let me walk you through how a change actually goes from a developer's laptop to running infrastructure. There are essentially 5 steps."

**Step 1 — Clone and code locally**
"We clone the repo to our local machine and make changes in VS Code. The Terraform files are just HCL — human-readable config files. You can see in the images I have the repo open in VS Code with the full folder tree visible."

**Step 2 — Push a branch and open a PR**
"Once changes are ready, we push to a new branch and open a Pull Request. We don't push directly to `main` — all changes go through PR."

**Step 3 — 4 automated checks run on the PR**
*(point at Image 3 — validation.yaml in VS Code)*

"When a PR is opened, GitHub Actions automatically runs 4 checks. One of them is the PR title validation you can see here — `validation.yaml`. This enforces conventional commit format on every PR title. The allowed types are: `fix`, `feat`, `docs`, `ci`, `chore`, `style`, `refactor`, `test`, `perf`. If your PR title doesn't match one of those prefixes, the check fails and you can't merge. This keeps the commit history clean and readable.

The other checks cover things like Terraform format validation and linting using `tflint` — you can see `.tflint.hcl` in the repo root.

One of the 4 checks is a webhook that triggers Terraform Cloud directly from the PR. At this stage Terraform Cloud runs a **plan only** — it calculates what would change but does not apply anything. The result of that plan comes back as a pass or fail status right on the PR. So before anyone approves the code, the team can already see whether the infrastructure change is valid and what it would do. Nothing is touched in AWS yet.

**Step 4 — Approval and merge**
"Once all 4 checks pass and the Terraform plan looks good, we request a review from an approved team member. Once they approve, we merge to `main`."

**Step 5 — Terraform Cloud applies on merge**
"The merge to `main` is what triggers the actual **plan and apply** in Terraform Cloud. This is the only moment Terraform makes real changes to AWS. We don't manually run `terraform apply` from anyone's laptop — the merge is the trigger. Terraform Cloud handles everything from there."

---

## TERRAFORM CLOUD — Two Workspaces

*(point at Images 4 and 5 — Terraform Cloud workspace pages)*

"We have two workspaces in Terraform Cloud, both under the `worldpay-tf` organization.

The first is `phoenix-statement-reporting-data-persistance-dev` — this manages 14 resources, all the storage layer. The second is `phoenix-statement-reporting-compute-dev` — this manages 28 resources, all the compute layer.

The reason `data-persistance` must run before `compute` is because `compute` reads `data-persistance` outputs directly using `tfe_outputs`. Things like the DynamoDB table ARN, the S3 bucket ARNs — these are all passed from one workspace to the other automatically. Terraform Cloud handles this dependency.

After a merge, you come here to see the run status. You want to see `Applied` in green — that means the plan ran, Terraform calculated what changed, and it applied successfully. If you see `Errored`, you click into the run, read the error, fix the code, push a new branch, go through the PR process again, and let it re-run. You never manually apply from your laptop — everything goes through this UI."

---

## THE OIDC REPO — How Terraform Cloud Talks to AWS

*(point at Images 7 and 8 — OIDC repo on GitHub)*

"There's a second repo involved — `wp-cloudops-aws-accounts-mgmt` in the `wp-cloud-services` org. This is a Worldpay-managed central repo.

For Terraform Cloud to be able to create resources in our AWS account, it needs an IAM access role in that account. This repo is where that role lives. We followed the same process — clone, make our change, push, PR, get approved, merge — and once merged, Terraform Cloud has the IAM access it needs to authenticate and deploy into our AWS account. That's all you need to know about that repo."

---

## CLOSING

"So that's the full workflow. Code lives in GitHub, CI checks run on every PR, merges trigger Terraform Cloud, and Terraform Cloud authenticates to AWS via OIDC. No manual applies, no stored credentials, no one running terraform from their laptop against prod.

Now let's go module by module and walk through what we actually provisioned and why."

---
