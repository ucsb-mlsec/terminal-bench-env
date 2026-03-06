terraform {
  required_version = ">= 1.5.0, < 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "production"
      ManagedBy   = "Terraform"
      Project     = "multi-env-infrastructure"
    }
  }
}

module "networking" {
  source = "../modules/networking"
  
  environment         = "prod"
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  public_subnets     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnets    = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

module "application" {
  source = "git@github.com:company/terraform-app-module.git//app?ref=v2.1.0"
  
  environment      = "prod"
  vpc_id          = module.networking.vpc_id
  private_subnets = module.networking.private_subnet_ids
  instance_type   = "t3.large"
  min_size        = 3
  max_size        = 10
}

module "database" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~>6.0"
  
  identifier = "prod-db"
  
  engine               = "postgres"
  engine_version       = "14.7"
  family              = "postgres14"
  major_engine_version = "14"
  instance_class       = "db.r5.xlarge"
  
  allocated_storage = 100
  
  db_name  = "production"
  username = var.db_username
  port     = 5432
  
  vpc_security_group_ids = [module.networking.database_security_group_id]
  db_subnet_group_name   = module.networking.database_subnet_group
  
  backup_retention_period = 30
  multi_az               = true
}

module "monitoring" {
  source = "../modules/observability"
  
  environment = "prod"
  vpc_id     = module.networking.vpc_id
  
  enable_cloudwatch = true
  enable_xray      = true
  log_retention    = 90
}

module "security" {
  source = "../modules/security"
  
  environment = "prod"
  vpc_id     = module.networking.vpc_id
  
  app_security_group_id = module.application.security_group_id
}

variable "aws_region" {
  description = "AWS region for production environment"
  type        = string
  default     = "us-east-1"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  sensitive   = true
}

output "vpc_id" {
  description = "Production VPC ID"
  value       = module.networking.vpc_id
}

output "application_endpoint" {
  description = "Application load balancer endpoint"
  value       = module.application.lb_endpoint
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value       = module.database.db_instance_endpoint
  sensitive   = true
}