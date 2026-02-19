variable "region" {
  type        = string
  description = "AWS region where resources will be deployed"
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Environment name for resource tagging and naming"
  default     = "dev"
}

variable "project_name" {
  type        = string
  description = "Name of the project for resource identification"
}

variable "instance_count" {
  type        = number
  description = "Number of instances to create"
  default     = 2
}

variable "enable_monitoring" {
  type        = bool
  description = "Enable CloudWatch monitoring for resources"
  default     = true
}

variable "tags" {
  type        = map(string)
  description = "Common tags to apply to all resources"
  default = {
    ManagedBy = "Terraform"
    Owner     = "DevOps"
    Project   = "Infrastructure"
  }
}