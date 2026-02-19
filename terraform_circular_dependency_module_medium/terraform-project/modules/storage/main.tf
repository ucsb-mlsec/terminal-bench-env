variable "security_group_id" {
  description = "Security group ID from network module"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID from network module"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs from network module"
  type        = list(string)
  default     = []
}

resource "aws_s3_bucket" "app_storage" {
  bucket_prefix = "app-storage-"
  
  tags = {
    Name        = "Application Storage"
    VPC         = var.vpc_id
    Environment = "production"
  }
}

resource "aws_s3_bucket_policy" "app_storage_policy" {
  bucket = aws_s3_bucket.app_storage.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowVPCAccess"
        Effect = "Allow"
        Principal = "*"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.app_storage.arn}/*"
        Condition = {
          StringEquals = {
            "aws:SourceVpc" = var.vpc_id
          }
        }
      }
    ]
  })
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id       = var.vpc_id
  service_name = "com.amazonaws.us-east-1.s3"
  
  tags = {
    Name            = "S3 VPC Endpoint"
    SecurityGroup   = var.security_group_id
    Environment     = "production"
  }
}

output "bucket_id" {
  description = "ID of the S3 bucket"
  value       = aws_s3_bucket.app_storage.id
}

output "bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.app_storage.arn
}

output "vpc_endpoint_id" {
  description = "ID of the S3 VPC endpoint"
  value       = aws_vpc_endpoint.s3.id
}