terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

module "network" {
  source            = "../modules/network"
  storage_bucket_id = module.storage.bucket_id
  environment       = "production"
}

module "storage" {
  source            = "../modules/storage"
  vpc_id            = module.network.vpc_id
  security_group_id = module.network.security_group_id
  environment       = "production"
}

output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.network.vpc_id
}

output "security_group_id" {
  description = "The ID of the security group"
  value       = module.network.security_group_id
}

output "bucket_id" {
  description = "The ID of the storage bucket"
  value       = module.storage.bucket_id
}

output "bucket_arn" {
  description = "The ARN of the storage bucket"
  value       = module.storage.bucket_arn
}