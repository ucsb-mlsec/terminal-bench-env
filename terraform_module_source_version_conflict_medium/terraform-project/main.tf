terraform {
  required_version = ">= 1.5.0"
}

# Provider configuration
provider "aws" {
  region = "us-west-2"
}

# Local networking module - BROKEN: missing quotes and wrong path format
module "networking" {
  source = modules/networking
  
  vpc_cidr = "10.0.0.0/16"
  environment = "production"
  region = "us-west-2"
}

# Local compute module - BROKEN: using backslashes instead of forward slashes
module "compute" {
  source = ".\modules\compute"
  
  instance_type = "t3.medium"
  instance_count = 3
  subnet_ids = module.networking.private_subnet_ids
}

# Git module - BROKEN: malformed URL with incorrect protocol syntax
module "security_groups" {
  source = "git::https//github.com/terraform-modules/aws-security-groups.git"
  
  vpc_id = module.networking.vpc_id
  environment = "production"
  allowed_cidr_blocks = ["10.0.0.0/8"]
}

# Registry module - BROKEN: missing equals sign in version constraint
module "s3_bucket" {
  source = "terraform-aws-modules/s3-bucket/aws"
  version "3.5.0"
  
  bucket_name = "my-application-bucket"
  versioning_enabled = true
  encryption_enabled = true
}

# Another local module - BROKEN: missing ./ prefix for relative path
module "database" {
  source = "modules/rds"
  
  db_name = "myapp"
  db_instance_class = "db.t3.large"
  allocated_storage = 100
  engine_version = "14.7"
  vpc_id = module.networking.vpc_id
}

# Registry module - BROKEN: incorrect registry format missing namespace
module "vpc_endpoints" {
  source = registry.terraform.io/vpc-endpoints/aws
  version = "2.1.0"
  
  vpc_id = module.networking.vpc_id
  endpoint_services = ["s3", "dynamodb"]
}