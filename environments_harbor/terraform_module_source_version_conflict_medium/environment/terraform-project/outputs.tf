output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.networking.vpc_id
}

output "subnet_ids" {
  description = "List of subnet IDs created in the VPC"
  value       = module.networking.subnet_ids
}

output "instance_ips" {
  description = "IP addresses of the compute instances"
  value       = module.compute.instance_ips
}

output "environment" {
  description = "The deployment environment name"
  value       = var.environment
}