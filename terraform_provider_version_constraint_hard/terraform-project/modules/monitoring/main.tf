terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.0, < 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 2.3"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.7"
    }
  }
  required_version = ">= 0.13"
}

variable "alarm_threshold" {
  description = "CloudWatch alarm threshold value"
  type        = number
  default     = 80
}

variable "retention_days" {
  description = "Log retention in days"
  type        = number
  default     = 30
}

variable "email_endpoint" {
  description = "Email address for SNS notifications"
  type        = string
}

resource "random_string" "topic_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "time_offset" "retention_calc" {
  offset_days = var.retention_days
}

resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/application/logs"
  retention_in_days = var.retention_days
}

resource "aws_cloudwatch_log_group" "system" {
  name              = "/aws/system/logs"
  retention_in_days = var.retention_days
}

resource "aws_cloudwatch_log_group" "security" {
  name              = "/aws/security/audit"
  retention_in_days = var.retention_days * 2
}

resource "aws_sns_topic" "alerts" {
  name = "cloudwatch-alerts-${random_string.topic_suffix.result}"
}

resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.email_endpoint
}

resource "aws_cloudwatch_metric_alarm" "cpu_utilization" {
  alarm_name          = "high-cpu-utilization"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = var.alarm_threshold
  alarm_description   = "Triggers when CPU exceeds threshold"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

output "log_group_names" {
  value = [
    aws_cloudwatch_log_group.application.name,
    aws_cloudwatch_log_group.system.name,
    aws_cloudwatch_log_group.security.name
  ]
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}