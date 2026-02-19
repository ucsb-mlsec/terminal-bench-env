variable "security_group_id" {
  type        = string
  description = "The security group ID from the network module to attach to storage resources"
}

variable "vpc_id" {
  type        = string
  description = "The VPC ID from the network module for storage network configuration"
}

variable "bucket_name_prefix" {
  type        = string
  default     = "app-storage"
  description = "Prefix for storage bucket names"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Environment name (dev, staging, prod)"
}