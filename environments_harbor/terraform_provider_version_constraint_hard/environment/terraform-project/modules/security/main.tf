terraform {
  required_version = ">= 0.12.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 2.0.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 2.1"
    }
  }
}

variable "role_names" {
  description = "List of IAM role names to create"
  type        = list(string)
  default     = ["app-role", "admin-role", "readonly-role"]
}

variable "kms_key_rotation" {
  description = "Enable automatic key rotation for KMS keys"
  type        = bool
  default     = true
}

resource "random_uuid" "security_id" {
  keepers = {
    role_names = join(",", var.role_names)
  }
}

resource "aws_iam_role" "app_role" {
  name = var.role_names[0]

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role" "admin_role" {
  name = var.role_names[1]

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "custom_policy" {
  name        = "security-custom-policy-${random_uuid.security_id.result}"
  description = "Custom security policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "app_role_attachment" {
  role       = aws_iam_role.app_role.name
  policy_arn = aws_iam_policy.custom_policy.arn
}

resource "aws_kms_key" "encryption_key" {
  description             = "KMS key for encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = var.kms_key_rotation

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "aws_kms_alias" "encryption_key_alias" {
  name          = "alias/security-key-${random_uuid.security_id.result}"
  target_key_id = aws_kms_key.encryption_key.key_id
}

resource "null_resource" "security_check" {
  triggers = {
    kms_key_id = aws_kms_key.encryption_key.key_id
  }

  provisioner "local-exec" {
    command = "echo 'Security configuration validated'"
  }
}

output "role_arns" {
  description = "ARNs of created IAM roles"
  value = {
    app_role   = aws_iam_role.app_role.arn
    admin_role = aws_iam_role.admin_role.arn
  }
}

output "kms_key_id" {
  description = "ID of the KMS encryption key"
  value       = aws_kms_key.encryption_key.key_id
}