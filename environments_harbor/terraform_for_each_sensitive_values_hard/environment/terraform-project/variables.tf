variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment"
}

variable "db_passwords" {
  type        = map(string)
  description = "Database passwords for each environment"
  sensitive   = true
  default = {
    dev        = "dev_pass_123"
    staging    = "staging_pass_456"
    production = "prod_pass_789"
  }
}

variable "admin_email" {
  type        = string
  default     = "admin@example.com"
  description = "Administrator email address"
}

variable "enable_monitoring" {
  type        = bool
  default     = true
  description = "Enable monitoring for database instances"
}