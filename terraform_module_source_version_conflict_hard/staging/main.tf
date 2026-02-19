terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "staging"
      ManagedBy   = "Terraform"
      Project     = "multi-env-infrastructure"
    }
  }
}

module "networking" {
  source  = "../modules/networking"
  version = ">= 2.0.0"

  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  availability_zones  = var.availability_zones
  enable_nat_gateway  = true
  enable_vpn_gateway  = false
}

module "application" {
  source = "git::https://github.com/example/terraform-modules.git//application?branch=main"

  environment     = var.environment
  vpc_id          = module.networking.vpc_id
  private_subnets = module.networking.private_subnet_ids
  instance_type   = var.app_instance_type
  min_size        = 2
  max_size        = 6
  desired_size    = 3
}

module "database" {
  source  = "terraform-aws/rds/aws"
  version = "~> 3.0"

  identifier        = "${var.environment}-database"
  engine            = "postgres"
  engine_version    = "13.7"
  instance_class    = var.db_instance_class
  allocated_storage = 100
  
  vpc_id             = module.networking.vpc_id
  subnet_ids         = module.networking.database_subnet_ids
  
  username = var.db_username
  password = var.db_password
}

module "logging" {
  source = "..\modules\logging"

  environment       = var.environment
  log_retention_days = 90
  enable_cloudwatch  = true
  enable_s3_export   = true
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "aws_region" {
  description = "AWS region for staging environment"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b"]
}

variable "app_instance_type" {
  description = "EC2 instance type for application servers"
  type        = string
  default     = "t3.medium"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.large"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}