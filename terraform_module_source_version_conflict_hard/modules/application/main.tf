terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

module "compute_instances" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "=> 5.0.0"
  
  count = var.instance_count
  
  name                   = "${var.environment}-app-${count.index}"
  instance_type          = var.instance_type
  ami                    = data.aws_ami.application.id
  subnet_id              = var.subnet_ids[count.index % length(var.subnet_ids)]
  vpc_security_group_ids = [aws_security_group.application.id]
  
  tags = merge(
    var.common_tags,
    {
      Name        = "${var.environment}-application-${count.index}"
      Environment = var.environment
      Component   = "application"
    }
  )
}

data "aws_ami" "application" {
  most_recent = true
  owners      = ["self"]
  
  filter {
    name   = "name"
    values = ["${var.app_name}-*"]
  }
  
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "application" {
  name        = "${var.environment}-app-sg"
  description = "Security group for application servers"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.load_balancer.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}

resource "aws_security_group" "load_balancer" {
  name        = "${var.environment}-lb-sg"
  description = "Security group for load balancer"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = var.common_tags
}