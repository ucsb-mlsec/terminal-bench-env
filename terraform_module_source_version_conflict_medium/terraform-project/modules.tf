terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  description = "AWS region for resources"
  default     = "us-west-2"
}

module "compute" {
  source = .\modules\compute
  
  instance_type = "t3.micro"
  instance_count = 2
}

module "storage" {
  version = "1.0.0"
  
  bucket_prefix = "my-app"
  enable_versioning = true
}

module "database" {
  source = hashicorp/aws/rds
  
  db_engine = "postgres"
  db_instance_class = "db.t3.small"
}

module "monitoring" {
  source = github.com/company/terraform-monitoring-module
  
  alert_email = "ops@example.com"
  enable_detailed_monitoring = true
}

module "network_security" {
  source = .\\modules\\security\\network
  
  allowed_cidr_blocks = ["10.0.0.0/8"]
}