variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string

  validation {
    condition     = can(regex("^([0-9]{1,3}.){3}[0-9]{1,3}/[0-9]{1,2}$", var.vpc_cidr))
    error_message = "VPC CIDR must be a valid CIDR block format (e.g., 10.0.0.0/16)."
  }
}

variable "aws_region" {
  description = "AWS region with availability zone suffix"
  type        = string

  validation {
    condition     = can(regex("(us|eu|ap|sa|ca|me|af)-(north|south|east|west|central)-[0-9][a-z]$", var.aws_region))
    error_message = "AWS region must be a valid region code with AZ suffix (e.g., us-east-1a)."
  }
}

variable "subnet_cidrs" {
  description = "List of subnet CIDR blocks"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for cidr in var.subnet_cidrs : can(regex("^([0-9]{1,3}\\.){3}[0-9]{1,3}/[0-9]{1,2}$", cidr))
    ])
    error_message = "All subnet CIDRs must be valid CIDR block formats."
  }
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in the VPC"
  type        = bool
  default     = true
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway for private subnets"
  type        = bool
  default     = false
}

variable "vpc_tags" {
  description = "Tags to apply to VPC resources"
  type        = map(string)
  default     = {}
}