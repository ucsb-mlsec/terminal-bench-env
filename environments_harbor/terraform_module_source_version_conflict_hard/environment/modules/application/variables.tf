# Application Module Variables
# Defines input variables for the application deployment module

variable "instance_type" {
  description = "The EC2 instance type for application servers"
  type        = string
  
  validation {
    condition     = can(regex("^t[23]\\.", var.instance_type))
    error_message = "Instance type must be a valid t2 or t3 instance family."
  }
}

variable "instance_count" {
  description = "Number of application instances to deploy"
  type        = number
  
  validation {
    condition     = var.instance_count > 0 && var.instance_count <= 10
    error_message = "Instance count must be between 1 and 10."
  }
}

variable "app_version" {
  description = "Version of the application to deploy"
  type        = string
  
  validation {
    condition     = can(regex("^v[0-9]+\\.[0-9]+\\.[0-9]+$", var.app_version))
    error_message = "App version must follow semantic versioning format (v1.0.0)."
  }
}

variable "network_id" {
  description = "VPC network ID where application resources will be deployed"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for distributing application instances"
  type        = list(string)
}