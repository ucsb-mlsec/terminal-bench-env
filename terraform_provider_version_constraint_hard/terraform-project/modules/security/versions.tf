terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      version = ">= 4.0, < 5.0"
    }
    random = {
      version = "~> 3.2"
    }
    null = {
      version = ">= 3.0"
    }
  }
}