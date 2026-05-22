# SCRIPT 03 — `core-networking` Module + Closing
### "Networking: Walking the Code Top to Bottom"

---

## OPENING

"The last module is `core-networking`. This one is shorter than the others — we started deploying networking infrastructure, hit some complexity around team alignment in the shared AWS account, and made a deliberate decision to pause. I'll walk through the code top to bottom and explain exactly what each piece does and where we stopped."

---

## LOCALS — Prefix

```hcl
locals {
  prefix = "wp-phoenix-statement-reporting"
}
```

"Same prefix as every other module — `wp-phoenix-statement-reporting`. All resource names start with this so everything belonging to our project is instantly identifiable in the AWS console."

---

## `aws_eip` — Elastic IP

```hcl
resource "aws_eip" "phoenix_nat_eip" {
  tags = {
    Name = "phoenix-nat-eip"
  }
}
```

"The first resource is an Elastic IP — a static public IP address. This is provisioned and active. The reason we need a static IP is that in financial services, outbound calls to external services often need to come from a known, whitelisted IP address. With a regular NAT Gateway your outbound IP can change; with an Elastic IP it's fixed. So this is the static outbound identity of our private network."

---

## `aws_nat_gateway` — NAT Gateway

```hcl
resource "aws_nat_gateway" "phoenix_nat_gw" {
  allocation_id = aws_eip.phoenix_nat_eip.id
  subnet_id     = "subnet-0ebb0938487245a85"  # phoenix-dev-public-us-east-2c
  tags = {
    Name = "phoenix-nat-gw"
  }
}
```

"The NAT Gateway sits in a public subnet in `us-east-2c` and is attached to the Elastic IP we just defined. This is also provisioned and running.

The purpose: resources in private subnets — like our SageMaker notebooks or Lambda functions if we move them into private subnets — need to make outbound calls to AWS services. The NAT Gateway is what allows that. Traffic goes: private subnet → NAT Gateway → Elastic IP → internet or AWS service endpoint. Inbound connections are blocked; only outbound is allowed through it.

The subnet ID is hardcoded here because this public subnet already existed in the account — it wasn't created by our Terraform. We're attaching the NAT Gateway to pre-existing infrastructure."

---

## COMMENTED-OUT RESOURCES — Private Route Tables

```hcl
# resource "aws_route" "private_2a_default" {
#   route_table_id         = "rtb-06825109b956ff247"
#   destination_cidr_block = "0.0.0.0/0"
#   nat_gateway_id         = aws_nat_gateway.phoenix_nat_gw.id
# }

# resource "aws_route" "private_2b_default" { ... }
# resource "aws_route" "private_2c_default" { ... }
```

"These three `aws_route` resources would update the private route tables for each availability zone — `us-east-2a`, `us-east-2b`, `us-east-2c` — to route all outbound traffic through the NAT Gateway. One route per AZ.

They are commented out deliberately. Here's why: we are in a shared AWS account. These route tables are shared infrastructure — other teams' resources also sit in these private subnets. If we push a default route change into those route tables, it affects everyone routing through them, not just our project. Before making that change we'd need to align with the other teams on what subnets they're using and confirm nothing would break.

We started that investigation, and it became clear that for a POC it wasn't worth the cross-team coordination overhead. So we paused. The NAT Gateway and EIP are live. When the time comes, uncommenting these three blocks, pushing a PR, and merging is all it takes to activate the routing. The heavy lifting is already done."

---

## CLOSING COMMENT IN THE FILE

```hcl
#git testing first push
```

"And there's a comment at the bottom — `git testing first push` — that's just a leftover from when this module was first committed. Not functional, just a trace of the first push."

---

## OVERALL CLOSING — What Terraform Gives Us

"Let me close the whole Terraform KT with a broader point on why we set it up this way.

The alternative to Terraform is clicking through the AWS console to provision resources. That works for experiments. But the moment you need to reproduce an environment — for a new developer, for a staging environment, for recovering from an incident — you want infrastructure as code. Every resource we've walked through today can be destroyed and recreated with a single run. Consistently. Without anyone needing to remember what they clicked.

Terraform Cloud adds the governance layer. Every infrastructure change goes through the same PR review process as application code. The run history is a full audit trail. The workspace separation means storage and compute are independently deployable.

For a POC, some of this feels like overhead. But when you hand this off or move to production, the infrastructure is documented, version controlled, and reviewable — not locked in someone's head."

---

## FULL RESOURCE SUMMARY

"To close, here's everything we provisioned:

**data-persistance workspace — 14 resources:**
- 4 S3 buckets: documents, Snowflake staging, Lambda packages, code backup
- 1 DynamoDB table with 2 GSIs
- 1 OpenSearch domain — vector database for embeddings
- Supporting IAM policies and CloudWatch logging

**compute workspace — 28 resources:**
- 1 Bedrock PII guardrail, versioned — actively used in the pipeline
- 3 IAM roles: SageMaker, Lambda, Step Functions — each scoped to what they need
- 3 Lambda functions: processor, hitl-wait, hitl-resolve
- 1 Step Functions state machine — the HITL waitForTaskToken flow
- 4 SageMaker notebooks — one per developer
- Supporting IAM policies

**core-networking — 2 active resources:**
- 1 Elastic IP
- 1 NAT Gateway — provisioned, pending route table alignment to activate routing

That's the full infrastructure footprint. Any questions?"

---
