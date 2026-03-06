# Logging Module Variables Configuration
# This file defines input variables for the centralized logging infrastructure module

variable "log_retention_days" {
  description = "Number of days to retain log data before archival or deletion"
  type        = number

  validation {
    condition     = var.log_retention_days >= 1 && var.log_retention_days <= 3650
    error_message = "Log retention days must be between 1 and 3650 days."
  }
}

variable "log_level" {
  description = "Logging level for application and system logs (DEBUG, INFO, WARN, ERROR)"
  type        = string

  validation {
    condition     = contains(["DEBUG", "INFO", "WARN", "ERROR"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARN, ERROR."
  }
}

variable "enable_encryption" {
  description = "Enable encryption at rest for log storage and transmission"
  type        = bool
  default     = true
}

variable "destination_buckets" {
  description = "List of S3 bucket names or storage destinations for log aggregation and archival"
  type        = list(string)

  validation {
    condition     = length(var.destination_buckets) > 0
    error_message = "At least one destination bucket must be specified for log storage."
  }
}