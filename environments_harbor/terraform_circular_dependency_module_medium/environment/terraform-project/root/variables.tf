variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region where resources will be deployed"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Environment name for resource tagging and naming"
}

variable "project_name" {
  type        = string
  default     = "terraform-refactor"
  description = "Project name used for resource naming and tagging"
}