# Database Module Variables

variable "db_engine" {
  description = "The database engine to use (e.g., postgres, mysql, mariadb)"
  type        = string
  
  validation {
    condition     = contains(["postgres", "mysql", "mariadb", "aurora-postgresql", "aurora-mysql"], var.db_engine)
    error_message = "Database engine must be one of: postgres, mysql, mariadb, aurora-postgresql, aurora-mysql."
  }
}

variable "db_version" {
  description = "The version of the database engine to use"
  type        = string
  default     = "14.7"
}

variable "instance_class" {
  description = "The instance class for the database (e.g., db.t3.micro, db.t3.small, db.r5.large)"
  type        = string
  default     = "db.t3.small"
}

variable "allocated_storage" {
  description = "The allocated storage in gigabytes for the database instance"
  type        = number
  default     = 20
  
  validation {
    condition     = var.allocated_storage >= 20 && var.allocated_storage <= 65536
    error_message = "Allocated storage must be between 20 and 65536 GB."
  }
}

variable "multi_az" {
  description = "Specifies if the database instance should be deployed in multiple availability zones for high availability"
  type        = bool
  default     = false
}