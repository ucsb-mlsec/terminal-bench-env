variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "ami_id" {
  description = "AMI ID for EC2 instances"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs"
  type        = list(string)
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

resource "random_pet" "instance_name" {
  length    = 2
  separator = "-"
}

resource "aws_security_group" "compute_sg" {
  name        = "compute-sg-${random_pet.instance_name.id}"
  description = "Security group for compute instances"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "compute-sg-${random_pet.instance_name.id}"
    Environment = var.environment
  }
}

resource "aws_instance" "compute" {
  count                  = 3
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = element(var.subnet_ids, count.index)
  vpc_security_group_ids = [aws_security_group.compute_sg.id]

  tags = {
    Name        = "${random_pet.instance_name.id}-instance-${count.index + 1}"
    Environment = var.environment
  }
}

resource "time_sleep" "wait_for_instances" {
  depends_on      = [aws_instance.compute]
  create_duration = "30s"
}

resource "null_resource" "post_deployment" {
  depends_on = [time_sleep.wait_for_instances]

  provisioner "local-exec" {
    command = "echo 'Deployment completed for instances'"
  }

  triggers = {
    instance_ids = join(",", aws_instance.compute[*].id)
  }
}

output "instance_ids" {
  description = "IDs of created EC2 instances"
  value       = aws_instance.compute[*].id
}

output "public_ips" {
  description = "Public IPs of EC2 instances"
  value       = aws_instance.compute[*].public_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.compute_sg.id
}