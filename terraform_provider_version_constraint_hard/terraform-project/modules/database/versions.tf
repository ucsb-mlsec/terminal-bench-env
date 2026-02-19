terraform {
  required_version = ">= 0.12, < 2.0"
}

provider "aws" {
  version = "~> 3.0"
}

provider "random" {
  version = "= 3.0.0"
}

provider "local" {
  version = "~> 1.4"
}