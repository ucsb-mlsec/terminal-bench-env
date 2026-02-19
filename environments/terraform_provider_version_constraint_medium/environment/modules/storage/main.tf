terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.50"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 2.0"
    }
    google = {
      source  = "hashicorp/google"
      version = ">= 3.0, < 5.0"
    }
  }
}

variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
}

variable "storage_tier" {
  description = "Storage tier for Azure storage account"
  type        = string
  default     = "Standard"
}

variable "location" {
  description = "Azure location for storage account"
  type        = string
  default     = "eastus"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name

  tags = {
    Name        = var.bucket_name
    Environment = "production"
  }
}

resource "azurerm_storage_account" "main" {
  name                     = replace(var.bucket_name, "-", "")
  resource_group_name      = "storage-rg"
  location                 = var.location
  account_tier             = var.storage_tier
  account_replication_type = "LRS"
}

resource "google_storage_bucket" "main" {
  name     = "${var.bucket_name}-gcp"
  location = "US"
  project  = var.project_id
}

output "s3_bucket_id" {
  description = "ID of the S3 bucket"
  value       = aws_s3_bucket.main.id
}

output "azure_storage_id" {
  description = "ID of the Azure storage account"
  value       = azurerm_storage_account.main.id
}

output "gcs_bucket_url" {
  description = "URL of the Google Cloud Storage bucket"
  value       = google_storage_bucket.main.url
}