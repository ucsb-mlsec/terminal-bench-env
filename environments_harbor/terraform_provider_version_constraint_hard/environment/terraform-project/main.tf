terraform {
  required_version = ">= 1.0.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

module "networking" {
  source = "./modules/networking"
  
  vpc_cidr = "10.0.0.0/16"
  environment = var.environment
}

module "compute" {
  source = "./modules/compute"
  
  vpc_id = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  environment = var.environment
}

module "database" {
  source = "./modules/database"
  
  vpc_id = module.networking.vpc_id
  subnet_ids = module.networking.database_subnet_ids
  environment = var.environment
}

module "monitoring" {
  source = "./modules/monitoring"
  
  resources_to_monitor = {
    compute_instances = module.compute.instance_ids
    database_clusters = module.database.cluster_ids
  }
}

module "security" {
  source = "./modules/security"
  
  vpc_id = module.networking.vpc_id
  environment = var.environment
}

output "vpc_id" {
  value = module.networking.vpc_id
}

output "compute_instance_ids" {
  value = module.compute.instance_ids
}

output "database_endpoint" {
  value = module.database.endpoint
}