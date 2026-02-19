#!/bin/bash
# Spark Structured Streaming Application Startup Script
# This script starts multiple Spark streaming applications with their respective checkpoint locations
# Purpose: Launch streaming jobs with proper checkpoint configuration for fault tolerance

export CHECKPOINT_BASE="/mnt/old_storage/checkpoints"
export APP_CHECKPOINT="/mnt/old_storage/checkpoints/app_main"
SPARK_HOME=/opt/spark
APP_JAR=/opt/streaming_app/lib/streaming-app.jar

# Start Main Streaming Application
echo "Starting Main Streaming Application..."
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.sql.streaming.checkpointLocation=/mnt/old_storage/checkpoints/spark_submit_app \
  --class com.example.StreamingApp \
  $APP_JAR

# Start Data Ingestion Stream
echo "Starting Data Ingestion Stream..."
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.sql.streaming.checkpointLocation=/mnt/old_storage/checkpoints/ingestion_stream \
  --conf spark.executor.memory=4g \
  --class com.example.IngestionStreamApp \
  $APP_JAR

# Start Analytics Aggregation Stream
echo "Starting Analytics Aggregation Stream..."
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --conf spark.sql.streaming.checkpointLocation=/mnt/old_storage/checkpoints/analytics_agg \
  --conf spark.executor.memory=8g \
  --conf spark.executor.cores=4 \
  --class com.example.AnalyticsAggregationApp \
  $APP_JAR

echo "All streaming applications started successfully"