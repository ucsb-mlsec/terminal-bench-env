output "vpc_id" {
  description = "The ID of the VPC created by the network module"
  value       = module.network.vpc_id
}

output "security_group_id" {
  description = "The ID of the security group for storage access"
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