variable "email_address" {
  description = "Email address for notification endpoints"
  type        = string

  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}$", var.email_address))
    error_message = "The email_address must be a valid email format (e.g., user@example.com, admin@company.org)."
  }
}