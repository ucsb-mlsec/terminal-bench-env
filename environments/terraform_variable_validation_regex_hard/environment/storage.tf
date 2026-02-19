variable "s3_bucket_name" {
  description = "Name for the S3 bucket, must follow AWS naming conventions"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]+[a-z0-9]$", var.s3_bucket_name)) && length(var.s3_bucket_name) >= 3 && length(var.s3_bucket_name) <= 63 && !can(regex("--", var.s3_bucket_name))
    error_message = "S3 bucket name must be 3-63 characters, contain only lowercase letters, numbers, and hyphens, cannot have consecutive hyphens, and cannot start or end with a hyphen."
  }
}