terraform {
  required_version = "= 1.5.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "= 5.12.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "= 3.5.1"
    }
    time = {
      source  = "hashicorp/time"
      version = "= 0.9.1"
    }
  }
}