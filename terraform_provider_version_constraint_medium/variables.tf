variable "aws_region" {
  type        = string
  description = "AWS region for resource deployment"
  default     = "us-east-1"
}

variable "azure_location" {
  type        = string
  description = "Azure location for resource deployment"
  default     = "eastus"
}

variable "gcp_project" {
  type        = string
  description = "GCP project ID for resource deployment"
}

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, production)"
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "tags" {
  type        = map(string)
  description = "Common tags to apply to all resources"
  default = {
    ManagedBy = "Terraform"
  }
}