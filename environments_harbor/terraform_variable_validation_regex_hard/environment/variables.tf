variable "environment_name" {
  description = "Environment name with specific format requirements"
  type        = string

  validation {
    condition     = can(regex("^(dev|staging|prod)-[a-z]+-[0-9]{2}", var.environment_name))
    error_message = "Environment name must follow pattern: (dev|staging|prod)-<app>-<number>, e.g., 'dev-app-01'."
  }
}

variable "version_string" {
  description = "Semantic version string for application versioning"
  type        = string

  validation {
    condition     = can(regex("^[0-9]+.[0-9]+.[0-9]+$", var.version_string))
    error_message = "Version string must follow semantic versioning format: X.Y.Z where X, Y, Z are numbers."
  }
}

variable "tag_value" {
  description = "Tag value for resource tagging with specific constraints"
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-]+$", var.tag_value)) && length(var.tag_value) < 256
    error_message = "Tag value must contain only alphanumeric characters, hyphens, and underscores, and be less than 256 characters."
  }
}

variable "iam_role_name" {
  description = "IAM role name following AWS naming conventions"
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9_-{1,64}$", var.iam_role_name))
    error_message = "IAM role name must be alphanumeric with hyphens and underscores, 1-64 characters long."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for VPC configuration"
  type        = string

  validation {
    condition     = can(regex("^([0-9]{1,3}.){3}[0-9]{1,3}/[0-9]{1,2}$", var.vpc_cidr))
    error_message = "VPC CIDR must be a valid CIDR block format (e.g., 10.0.0.0/16)."
  }
}

variable "aws_region" {
  description = "AWS region code with optional availability zone suffix"
  type        = string

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]{1}[a-z]?$", var.aws_region))
    error_message = "AWS region must be a valid region code (e.g., us-east-1, us-east-1a)."
  }
}

variable "s3_bucket_name" {
  description = "S3 bucket name following AWS naming conventions"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]+[a-z0-9-]*[a-z0-9]+$", var.s3_bucket_name)) && length(var.s3_bucket_name) >= 3 && length(var.s3_bucket_name) <= 63 && !can(regex("--", var.s3_bucket_name))
    error_message = "S3 bucket name must be lowercase alphanumeric with hyphens, 3-63 characters, no consecutive hyphens."
  }
}

variable "notification_email" {
  description = "Email address for notification endpoints"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}$", var.notification_email))
    error_message = "Notification email must be a valid email address format."
  }
}