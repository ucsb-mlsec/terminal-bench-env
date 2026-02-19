terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for development environment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "app_instance_type" {
  description = "Instance type for application servers"
  type        = string
  default     = "t3.micro"
}

module "networking" {
  source = ..modules/networking
  
  environment = var.environment
  vpc_cidr    = var.vpc_cidr
}

module "application" {
  source = "git::http://github.com/company/terraform-app-module"
  
  environment     = var.environment
  vpc_id          = module.networking.vpc_id
  subnet_ids      = module.networking.private_subnet_ids
  instance_type   = var.app_instance_type
}

module "database" {
  source = "terraform-aws-modules/rds/aws"
  
  identifier     = "${var.environment}-database"
  engine         = "postgres"
  engine_version = "14.7"
  instance_class = "db.t3.micro"
  
  vpc_security_group_ids = [module.networking.database_sg_id]
  subnet_ids             = module.networking.database_subnet_ids
}

module "monitoring" {
  source  = "cloudposse/cloudwatch-logs/aws"
  version = ">> 0.6.0"
  
  namespace   = var.environment
  stage       = var.environment
  name        = "monitoring"
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "app_endpoint" {
  description = "Application endpoint"
  value       = module.application.endpoint
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = module.database.endpoint
}