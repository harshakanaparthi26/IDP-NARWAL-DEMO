module "hcptf-wp-phoenix-statement-reporting-123456" {
  source  = "app.terraform.io/worldpay-tf/iam-oidc-provider-and-role/aws"
  version = "1.0.0"

  oidc_provider_url    = "https://app.terraform.io"
  oidc_role            = "hcptf-wp-phoenix-statement-reporting-123456"
  custom_policy        = "hcptf-wp-phoenix-statement-reporting-123456-policy-1"
  max_session_duration = 43200
  create_oidc_provider = false

  policy_json = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [

    {
      "Sid": "FullServiceAccess",
      "Effect": "Allow",
      "Action": [
        "s3:*",
        "cloudwatch:*",
        "logs:*",
        "dynamodb:DescribeTable",
        "dynamodb:*",
        "lambda:*"
      ],
      "Resource": [
        "arn:aws:s3:::wp-phoenix-statement-reporting-*",
        "arn:aws:s3:::wp-phoenix-statement-reporting-*/*",
        "arn:aws:cloudwatch:*:*:wp-phoenix-statement-reporting-*",
        "arn:aws:logs:*:*:wp-phoenix-statement-reporting-*",
        "arn:aws:dynamodb:us-east-2:566603766408:table/wp-phoenix-statement-reporting-table",
        "arn:aws:lambda:*:566603766408:function:wp-phoenix-statement-reporting-*"
      ]
    },

    {
      "Sid": "CloudWatchLogsListDescribe",
      "Effect": "Allow",
      "Action": [
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams",
        "logs:DescribeResourcePolicies"
      ],
      "Resource": "*"
    },

    {
      "Sid": "AllowDeleteResourcePolicy",
      "Effect": "Allow",
      "Action": "logs:DeleteResourcePolicy",
      "Resource": "*"
    },

    {
      "Sid": "BedrockFullServiceAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:*"
      ],
      "Resource": "*"
    },

    {
      "Sid": "AllowPutResourcePolicy",
      "Effect": "Allow",
      "Action": "logs:PutResourcePolicy",
      "Resource": "*"
    }

  ]
}
EOF

  assume_role_conditions = [
    {
      test     = "StringEquals"
      variable = "app.terraform.io:aud"
      values   = ["aws.workload.identity"]
    },
    {
      test     = "StringLike"
      variable = "app.terraform.io:sub"
      values = [
        for value in local.tfc_subject :
        "organization/worldpay-tf/project:${local.tfc_project}:$${value}"
      ]
    }
  ]
}

# ------------------------------------------------------------------------------
# Additional IAM policies attached to the same role
# ------------------------------------------------------------------------------

resource "aws_iam_policy" "statement_reporting_policy_2" {
  name = "wp-phoenix-statement-reporting-policy-2"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "IAMPermissions"
        Effect = "Allow"
        Action = [
          "iam:TagOpenIDConnectProvider",
          "iam:CreateOpenIDConnectProvider",
          "iam:List*",
          "iam:Get*",
          "iam:TagPolicy",
          "iam:TagRole",
          "iam:TagInstanceProfile",
          "iam:GenerateCredentialReport",
          "iam:GenerateServiceLastAccessedDetails",
          "iam:GenerateOrganizationsAccessReport",
          "iam:SimulateCustomPolicy",
          "iam:SimulatePrincipalPolicy",
          "iam:CreateInstanceProfile",
          "iam:DeleteInstanceProfile",
          "iam:CreateServiceLinkedRole",
          "iam:SetDefaultPolicyVersion",
          "iam:GetRolePolicy",
          "iam:CreateRole",
          "iam:ListRolePolicies",
          "iam:PutRolePolicy",
          "iam:CreatePolicy",
          "iam:DeletePolicy",
          "iam:DeleteRolePolicy",
          "iam:GetPolicy",
          "iam:GetPolicyVersion",
          "iam:ListPolicyVersions",
          "iam:ListRoles",
          "iam:ListPolicyTags",
          "iam:ListPolicies",
          "iam:ListRoleTags",
          "iam:ListUserPolicies",
          "iam:ListAttachedRolePolicies",
          "iam:AttachRolePolicy",
          "iam:CreatePolicyVersion",
          "iam:ListEntitiesForPolicy",
          "iam:DetachRolePolicy",
          "iam:ListInstanceProfilesForRole",
          "iam:DeleteRole",
          "iam:UpdateRoleDescription",
          "iam:DeletePolicyVersion"
        ]
        Resource = [
          "arn:aws:iam::566603766408:role/wp-phoenix-statement-reporting-*",
          "arn:aws:iam::566603766408:policy/wp-phoenix-statement-reporting-*"
        ]
      },
      {
        Sid    = "RoleManagement"
        Effect = "Allow"
        Action = [
          "iam:UpdateAssumeRolePolicy",
          "iam:RemoveRoleFromInstanceProfile",
          "iam:AddRoleToInstanceProfile",
          "iam:UpdateOpenIDConnectProviderThumbprint",
          "iam:DeleteOpenIDConnectProvider",
          "iam:UpdateRole"
        ]
        Resource = [
          "arn:aws:iam::566603766408:role/wp-phoenix-statement-reporting-*",
          "arn:aws:iam::566603766408:policy/wp-phoenix-statement-reporting-*"
        ]
      },
      {
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          "arn:aws:iam::566603766408:role/wp-phoenix-statement-reporting-*"
        ]
      },
      {
        Sid    = "IAMCreateServiceLinkedRoleForEC2RelatedServices"
        Effect = "Allow"
        Action = "iam:CreateServiceLinkedRole"
        Resource = "*"
        Condition = {
          StringEquals = {
            "iam:AWSServiceName" = [
              "autoscaling.amazonaws.com",
              "ec2scheduled.amazonaws.com",
              "elasticloadbalancing.amazonaws.com",
              "spot.amazonaws.com",
              "spotfleet.amazonaws.com",
              "transitgateway.amazonaws.com"
            ]
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "statement_reporting_policy_3" {
  name = "wp-phoenix-statement-reporting-policy-3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerNotebookInstanceManagement"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateNotebookInstance",
          "sagemaker:DeleteNotebookInstance",
          "sagemaker:StartNotebookInstance",
          "sagemaker:StopNotebookInstance",
          "sagemaker:UpdateNotebookInstance",
          "sagemaker:DescribeNotebookInstance",
          "sagemaker:CreateNotebookInstanceLifecycleConfig",
          "sagemaker:UpdateNotebookInstanceLifecycleConfig",
          "sagemaker:DeleteNotebookInstanceLifecycleConfig",
          "sagemaker:DescribeNotebookInstanceLifecycleConfig",
          "sagemaker:ListNotebookInstances",
          "sagemaker:CreatePresignedNotebookInstanceUrl",
          "sagemaker:AddTags",
          "sagemaker:DeleteTags",
          "sagemaker:ListTags"
        ]
        Resource = [
          "arn:aws:sagemaker:*:566603766408:notebook-instance/wp-phoenix-statement-reporting-*",
          "arn:aws:sagemaker:*:566603766408:notebook-instance-lifecycle-config/wp-phoenix-statement-reporting-*"
        ]
      },
      {
        Sid    = "SageMakerNotebookReadListPresigned"
        Effect = "Allow"
        Action = [
          "sagemaker:ListNotebookInstances",
          "sagemaker:ListNotebookInstanceLifecycleConfigs",
          "sagemaker:CreatePresignedNotebookInstanceUrl"
        ]
        Resource = "*"
      },
      {
        Sid    = "OpenSearchFullAccess"
        Effect = "Allow"
        Action = [
          "es:ESHttpGet",
          "es:ESHttpPut",
          "es:ESHttpPost",
          "es:ESHttpDelete",
          "es:AddTags",
          "es:CreateDomain",
          "es:DeleteDomain",
          "es:DescribeDomain",
          "es:DescribeDomains",
          "es:DescribeDomainConfig",
          "es:UpdateDomainConfig",
          "es:ListDomainNames",
          "es:ListTags"
        ]
        Resource = [
          "arn:aws:es:*:566603766408:domain/wp-phoenix-statement-reporting-*",
          "arn:aws:es:*:566603766408:domain/wp-phoenix-opensearch-db"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "statement_reporting_policy_4" {
  name = "wp-phoenix-statement-reporting-policy-4"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "LambdaManagement"
        Effect = "Allow"
        Action = [
          "lambda:CreateFunction",
          "lambda:UpdateFunctionCode",
          "lambda:UpdateFunctionConfiguration",
          "lambda:DeleteFunction",
          "lambda:GetFunction",
          "lambda:ListFunctions",
          "lambda:InvokeFunction",
          "lambda:AddPermission",
          "lambda:TagResource",
          "lambda:UntagResource",
          "lambda:RemovePermission"
        ]
        Resource = [
          "arn:aws:lambda:*:566603766408:function:wp-phoenix-statement-reporting-*"
        ]
      },
      {
        Sid    = "ApiGatewayManagement"
        Effect = "Allow"
        Action = [
          "apigateway:POST",
          "apigateway:PUT",
          "apigateway:PATCH",
          "apigateway:DELETE",
          "apigateway:GET"
        ]
        Resource = "arn:aws:apigateway:*::/restapis/*"
      },
      {
        Sid      = "EC2FullAccess"
        Effect   = "Allow"
        Action   = "ec2:*"
        Resource = "*"
      },
      {
        Sid      = "ELBFullAccess"
        Effect   = "Allow"
        Action   = "elasticloadbalancing:*"
        Resource = "*"
      },
      {
        Sid      = "CloudWatchFullAccess"
        Effect   = "Allow"
        Action   = "cloudwatch:*"
        Resource = "*"
      },
      {
        Sid      = "AutoScalingFullAccess"
        Effect   = "Allow"
        Action   = "autoscaling:*"
        Resource = "*"
      },
      {
        Sid    = "StepFunctionsManageStateMachines"
        Effect = "Allow"
        Action = [
          "states:CreateStateMachine",
          "states:UpdateStateMachine",
          "states:ValidateStateMachineDefinition",
          "states:DeleteStateMachine",
          "states:ListStateMachineVersions",
          "states:DescribeStateMachine",
          "states:ListStateMachines"
        ]
        Resource = "arn:aws:states:us-east-2:566603766408:stateMachine:*"
      },
      {
        Sid    = "StepFunctionsTagging"
        Effect = "Allow"
        Action = [
          "states:TagResource",
          "states:UntagResource"
        ]
        Resource = "*"
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# Attach the additional policies to the role created by the module
# NOTE: The module manages policy-1 via custom_policy. Policies 2-4 are
#       attached here manually after the role is created.
# ------------------------------------------------------------------------------

resource "aws_iam_role_policy_attachment" "statement_reporting_policy_2" {
  role       = module.hcptf-wp-phoenix-statement-reporting-123456.role_name
  policy_arn = aws_iam_policy.statement_reporting_policy_2.arn
}

resource "aws_iam_role_policy_attachment" "statement_reporting_policy_3" {
  role       = module.hcptf-wp-phoenix-statement-reporting-123456.role_name
  policy_arn = aws_iam_policy.statement_reporting_policy_3.arn
}

resource "aws_iam_role_policy_attachment" "statement_reporting_policy_4" {
  role       = module.hcptf-wp-phoenix-statement-reporting-123456.role_name
  policy_arn = aws_iam_policy.statement_reporting_policy_4.arn
}
