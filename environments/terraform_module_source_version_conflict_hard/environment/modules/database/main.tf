terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

module "rds_instance" {
  source  = "terraform-aws-modules/rds/aws"
  version = ~> 5.0

  identifier = var.db_identifier

  engine               = "postgres"
  engine_version       = "14.7"
  family               = "postgres14"
  major_engine_version = "14"
  instance_class       = var.instance_class

  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage

  db_name  = var.database_name
  username = var.master_username
  port     = 5432

  multi_az               = var.multi_az
  db_subnet_group_name   = aws_db_subnet_group.database.name
  vpc_security_group_ids = [aws_security_group.database.id]

  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  create_cloudwatch_log_group     = true

  skip_final_snapshot       = var.environment == "dev"
  deletion_protection       = var.environment == "prod"
  apply_immediately         = var.environment == "dev"

  tags = var.tags
}

resource "aws_db_subnet_group" "database" {
  name       = "${var.db_identifier}-subnet-group"
  subnet_ids = var.database_subnet_ids

  tags = merge(var.tags, {
    Name = "${var.db_identifier}-subnet-group"
  })
}

resource "aws_security_group" "database" {
  name        = "${var.db_identifier}-sg"
  description = "Security group for ${var.db_identifier} database"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_groups
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.db_identifier}-sg"
  })
}

data "aws_db_instance" "existing" {
  count                  = var.use_existing_db ? 1 : 0
  db_instance_identifier = var.existing_db_identifier
}