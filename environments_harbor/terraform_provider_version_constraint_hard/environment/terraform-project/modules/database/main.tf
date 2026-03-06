resource "aws_db_subnet_group" "main" {
  name       = "${var.db_name}-subnet-group"
  subnet_ids = var.subnet_ids

  tags = {
    Name = "${var.db_name}-subnet-group"
  }
}

resource "aws_db_parameter_group" "main" {
  name   = "${var.db_name}-params"
  family = var.db_engine == "postgres" ? "postgres14" : "mysql8.0"

  parameter {
    name  = "max_connections"
    value = "100"
  }

  tags = {
    Name = "${var.db_name}-parameter-group"
  }
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

resource "aws_db_instance" "main" {
  identifier             = var.db_name
  engine                 = var.db_engine
  engine_version         = var.db_engine == "postgres" ? "14.7" : "8.0.32"
  instance_class         = var.db_instance_class
  allocated_storage      = var.allocated_storage
  storage_type           = "gp2"
  db_subnet_group_name   = aws_db_subnet_group.main.name
  parameter_group_name   = aws_db_parameter_group.main.name
  username               = var.db_username
  password               = random_password.db_password.result
  skip_final_snapshot    = true
  publicly_accessible    = false

  tags = {
    Name = var.db_name
  }
}

resource "local_file" "connection_string" {
  content  = "postgresql://${var.db_username}:${random_password.db_password.result}@${aws_db_instance.main.endpoint}/${var.db_name}"
  filename = "${path.module}/connection_string.txt"
}

variable "db_name" {
  type = string
}

variable "db_engine" {
  type    = string
  default = "postgres"
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "allocated_storage" {
  type    = number
  default = 20
}

variable "subnet_ids" {
  type = list(string)
}

variable "db_username" {
  type    = string
  default = "admin"
}

output "db_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "db_name" {
  value = aws_db_instance.main.identifier
}