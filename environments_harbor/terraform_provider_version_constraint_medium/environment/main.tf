terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 2.0, < 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = "us-east-1"
  default_tags {
    tags = {
      Environment = "production"
      ManagedBy   = "terraform"
    }
  }
}

provider "azurerm" {
  features {}
  skip_provider_registration = true
}

provider "google" {
  project = "my-project-id"
  region  = "us-central1"
}

module "compute" {
  source = "./modules/compute"
  
  environment = "production"
  instance_count = 3
}

module "networking" {
  source = "./modules/networking"
  
  vpc_cidr = "10.0.0.0/16"
  environment = "production"
}

module "storage" {
  source = "./modules/storage"
  
  bucket_prefix = "myapp"
  enable_versioning = true
}

resource "aws_s3_bucket" "main" {
  bucket = "myapp-main-bucket-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "Main Application Bucket"
    Environment = "production"
  }
}

resource "azurerm_resource_group" "main" {
  name     = "myapp-resources"
  location = "East US"
  
  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

output "s3_bucket_name" {
  value = aws_s3_bucket.main.id
}

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}