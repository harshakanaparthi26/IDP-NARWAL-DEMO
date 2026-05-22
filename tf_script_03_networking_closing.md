# SCRIPT 03 — `core-networking` Module + Closing
### "Networking: What We Tried, Why We Paused, What Comes Next"

---

## OPENING

"The last module is `core-networking`. This one is a bit different from the others — I'll be upfront about it. We started deploying networking infrastructure, hit some complexity around team alignment, and made the decision to pause rather than risk impacting other teams. This is actually a good example of responsible infrastructure management in a shared AWS account."

---

## WHAT'S DEPLOYED — NAT GATEWAY + ELASTIC IP

"Two resources are active in this module: an Elastic IP and a NAT Gateway.

```hcl
resource "aws_eip" "phoenix_nat_eip" { ... }

resource "aws_nat_gateway" "phoenix_nat_gw" {
  allocation_id = aws_eip.phoenix_nat_eip.id
  subnet_id     = "subnet-0ebb0938487245a85"  # phoenix-dev-public-us-east-2c
}
```

The NAT Gateway sits in a public subnet in `us-east-2c`. The Elastic IP gives it a static outbound IP address. The purpose of this setup is to allow resources in private subnets — like our SageMaker notebooks or Lambda functions, if we move them into a private subnet — to make outbound calls to AWS services without being directly exposed to the internet. Inbound connections are blocked; only outbound is allowed.

Why a static IP? In a financial services environment, outbound API calls to external services often need to come from a known, whitelisted IP. With a NAT Gateway + Elastic IP, all traffic from private subnets routes through one predictable IP address. That's the long-term goal.

The NAT Gateway is provisioned and running. The Elastic IP is allocated. But right now, nothing is routing through it."

---

## WHAT'S COMMENTED OUT — PRIVATE ROUTE TABLES

"If you look at the code, you'll see three `aws_route` resources are commented out:

```hcl
# resource "aws_route" "private_2a_default" { ... }
# resource "aws_route" "private_2b_default" { ... }
# resource "aws_route" "private_2c_default" { ... }
```

These would update the private route tables for all 3 availability zones — `us-east-2a`, `us-east-2b`, `us-east-2c` — to send all outbound traffic through the NAT Gateway.

The reason they're commented out: we're operating in a shared AWS account. The route tables and subnets in this account are shared infrastructure. If we updated those route tables, it could affect other teams' resources that also run in those private subnets. Before making a change that affects shared routing, you need to align with the other teams using the same account — understand what subnets they're using, confirm their traffic won't be disrupted.

We started that investigation, but aligning across teams takes time, and since this is a POC, we made the call to not block progress on it. The NAT Gateway is there waiting. Once we've confirmed the route table changes are safe, we uncomment those 3 blocks, push a PR, and it's done. The hard part — provisioning the NAT and EIP — is already complete."

---

## WHY THIS MATTERS FOR THE FUTURE

"This is worth talking about because it represents a real production requirement, not just a nice-to-have.

Right now, our SageMaker notebooks and Lambda functions are not inside a private subnet. They reach AWS services like Textract, Comprehend, Bedrock, and DynamoDB over the public internet using IAM credentials. That works for a POC, but in production you'd want all of this traffic to stay inside the AWS network using VPC endpoints — no public internet exposure at all.

The full production networking picture would look like this: SageMaker notebooks and Lambdas in private subnets, VPC endpoints for all AWS services they call, NAT Gateway only for any traffic that genuinely needs to go outbound to the internet, and route tables updated to enforce that topology.

We've started that journey with the NAT Gateway. The remaining work is the route table alignment, the VPC endpoint configuration, and moving the compute resources into the private subnets."

---

## OVERALL CLOSING — What Terraform Gives Us

"Let me close with a broader point about why we're using Terraform and Terraform Cloud for this at all.

The alternative would be clicking through the AWS Console to provision resources. That works for one-off experiments. But the moment you need to reproduce an environment — for a second developer, for a staging environment, for disaster recovery — you want infrastructure as code. Every resource we've walked through today can be destroyed and recreated with a single `terraform apply`. Consistently. Repeatably. Without someone needing to remember which checkboxes they clicked.

Terraform Cloud adds the collaboration and governance layer on top of that. Every infrastructure change goes through the same PR review process as application code. The run history gives us a full audit trail of what changed, when, and who triggered it. And the workspace separation means the storage layer and compute layer are independently deployable — if we need to update a Lambda, we don't have to touch the DynamoDB config.

For a POC, some of this might feel like overhead. But the discipline pays off when you hand this off to another team or move to production. The infrastructure is documented in code, version controlled, and reviewable — not locked in someone's head or buried in the console."

---

## SUMMARY — What We Provisioned

"To recap everything in Terraform for this project:

**data-persistance workspace (14 resources):**
- 4 S3 buckets: documents, Snowflake staging, Lambda packages, code backup
- 1 DynamoDB table with 2 GSIs — single table design
- 1 OpenSearch domain — vector database for embeddings
- Supporting IAM policies and CloudWatch logging

**compute workspace (28 resources):**
- 1 Bedrock guardrail with PII protection for 5 entity types, versioned
- 3 IAM roles: SageMaker, Lambda, Step Functions — each with scoped permissions
- 3 Lambda functions: processor, hitl-wait, hitl-resolve
- 1 Step Functions state machine — the HITL waitForTaskToken flow
- 4 SageMaker notebook instances — one per developer
- Supporting IAM policies wiring everything together

**core-networking (2 active resources):**
- 1 Elastic IP
- 1 NAT Gateway — ready to route, pending route table alignment

That's the full infrastructure footprint. Questions welcome."

---
