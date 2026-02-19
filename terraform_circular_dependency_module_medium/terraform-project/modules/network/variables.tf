variable "storage_bucket_id" {
  type        = string
  description = "The storage bucket ID from the storage module used for network access rules"
}

variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for the VPC network"
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Environment name for resource tagging"
}