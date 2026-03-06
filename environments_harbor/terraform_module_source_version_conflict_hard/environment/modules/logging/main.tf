terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "retention_days" {
  description = "Log retention in days"
  type        = number
  default     = 30
}

module "log_storage" {
  source = "..\shared\logs"
  
  environment     = var.environment
  log_bucket_name = "logs-${var.environment}"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

resource "aws_s3_bucket" "logs" {
  bucket = "application-logs-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Environment = var.environment
    Purpose     = "Application Logs"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs_lifecycle" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "log-retention"
    status = "Enabled"

    expiration {
      days = var.retention_days
    }
  }
}

resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/application/${var.environment}"
  retention_in_days = var.retention_days
}