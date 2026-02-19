variable "alert_email" {
  description = "Email address for receiving monitoring alerts and notifications"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "The alert_email must be a valid email address."
  }
}

variable "retention_days" {
  description = "Number of days to retain monitoring metrics and logs"
  type        = number
  default     = 30

  validation {
    condition     = var.retention_days >= 1 && var.retention_days <= 365
    error_message = "Retention days must be between 1 and 365."
  }
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed monitoring with high-resolution metrics (may incur additional costs)"
  type        = bool
}

variable "metric_namespaces" {
  description = "List of metric namespaces to monitor (e.g., AWS/EC2, AWS/RDS, Custom/Application)"
  type        = list(string)

  validation {
    condition     = length(var.metric_namespaces) > 0
    error_message = "At least one metric namespace must be specified."
  }
}