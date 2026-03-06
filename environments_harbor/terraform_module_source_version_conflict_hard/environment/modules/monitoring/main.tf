terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

module "monitoring_infrastructure" {
  source = "git::https://github.com/company/terrafrom-monitoring-module.git?ref=v2.1.0"
  
  environment     = var.environment
  project_name    = var.project_name
  alert_endpoints = var.alert_endpoints
  retention_days  = var.retention_days
}

module "metrics_collection" {
  source  = "hashicorp/metrics/aws"
  version = "=> 3.2.0"
  
  namespace          = var.metrics_namespace
  collection_interval = var.collection_interval
  enable_detailed_monitoring = true
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.environment}-monitoring-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EC2", "CPUUtilization"],
            ["AWS/RDS", "DatabaseConnections"]
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "Resource Utilization"
        }
      }
    ]
  })
}

resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "${var.environment}-high-cpu-usage"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = var.alarm_actions
}