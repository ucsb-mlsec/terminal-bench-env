#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time Log Processing Streaming Application
Processes log data from Kafka and aggregates metrics for dashboard display
"""

import json
import sys
import signal
from datetime import datetime
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

# ==================== CONFIGURATION PARAMETERS ====================
KAFKA_BROKERS = "localhost:9092"
KAFKA_TOPIC = "application-logs"
KAFKA_GROUP_ID = "log-processor-group"
CHECKPOINT_DIR = "/opt/spark_app/checkpoint"
OUTPUT_PATH = "/opt/spark_app/output/dashboard_data"

# Batch interval in seconds - THIS IS THE KEY TUNING PARAMETER
batch_interval = 10

# Application configuration
APP_NAME = "LogStreamProcessor"
WINDOW_DURATION = 60  # Window for aggregations in seconds
SLIDE_INTERVAL = 30   # Slide interval in seconds

# ==================== HELPER FUNCTIONS ====================

def parse_log_line(line):
    """
    Parse a JSON log line into structured data
    Expected format: {"timestamp": "...", "level": "...", "service": "...", "message": "...", "response_time": 123}
    """
    try:
        log_data = json.loads(line)
        return (
            log_data.get("service", "unknown"),
            {
                "level": log_data.get("level", "INFO"),
                "timestamp": log_data.get("timestamp", ""),
                "response_time": int(log_data.get("response_time", 0)),
                "message": log_data.get("message", "")
            }
        )
    except Exception as e:
        # Return a default tuple for malformed logs
        return ("unknown", {"level": "ERROR", "timestamp": "", "response_time": 0, "message": str(e)})

def filter_error_logs(log_tuple):
    """
    Filter to only process ERROR and CRITICAL level logs
    """
    service, log_data = log_tuple
    return log_data["level"] in ["ERROR", "CRITICAL", "WARN"]

def extract_service_metrics(log_tuple):
    """
    Extract key metrics per service for aggregation
    Returns: (service, (count, total_response_time, error_count))
    """
    service, log_data = log_tuple
    error_count = 1 if log_data["level"] in ["ERROR", "CRITICAL"] else 0
    return (service, (1, log_data["response_time"], error_count))

def aggregate_metrics(metrics1, metrics2):
    """
    Combine metrics from multiple records
    """
    count1, response_time1, errors1 = metrics1
    count2, response_time2, errors2 = metrics2
    return (count1 + count2, response_time1 + response_time2, errors1 + errors2)

def calculate_service_stats(service_metrics):
    """
    Calculate final statistics for each service
    Returns: (service, stats_dict)
    """
    service, metrics = service_metrics
    count, total_response_time, error_count = metrics
    avg_response_time = total_response_time / count if count > 0 else 0
    error_rate = (error_count / count * 100) if count > 0 else 0
    
    stats = {
        "service": service,
        "total_requests": count,
        "avg_response_time_ms": round(avg_response_time, 2),
        "error_count": error_count,
        "error_rate_percent": round(error_rate, 2),
        "timestamp": datetime.now().isoformat()
    }
    return (service, stats)

def save_to_dashboard(rdd):
    """
    Save aggregated results to output for dashboard consumption
    """
    try:
        if not rdd.isEmpty():
            results = rdd.collect()
            output_data = [stats for _, stats in results]
            
            # In production, this would write to a database or real-time dashboard
            # For this application, we write to a JSON file
            with open(f"{OUTPUT_PATH}/metrics_{datetime.now().timestamp()}.json", "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"[{datetime.now()}] Saved {len(output_data)} service metrics to dashboard")
    except Exception as e:
        print(f"Error saving to dashboard: {e}", file=sys.stderr)

def update_state_function(new_values, running_count):
    """
    Update function for stateful processing
    Maintains running totals across batches
    """
    if running_count is None:
        running_count = (0, 0, 0)
    
    for value in new_values:
        running_count = aggregate_metrics(running_count, value)
    
    return running_count

# ==================== MAIN APPLICATION ====================

def create_streaming_context():
    """
    Create and configure the Spark Streaming context
    """
    # Configure Spark
    conf = SparkConf().setAppName(APP_NAME)
    conf.set("spark.streaming.backpressure.enabled", "true")
    conf.set("spark.streaming.kafka.maxRatePerPartition", "1000")
    conf.set("spark.streaming.stopGracefullyOnShutdown", "true")
    
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    # Create streaming context with configured batch interval
    ssc = StreamingContext(sc, batch_interval)
    ssc.checkpoint(CHECKPOINT_DIR)
    
    return ssc

def setup_kafka_stream(ssc):
    """
    Setup Kafka direct stream for consuming log data
    """
    kafka_params = {
        "metadata.broker.list": KAFKA_BROKERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "largest"
    }
    
    kafka_stream = KafkaUtils.createDirectStream(
        ssc,
        [KAFKA_TOPIC],
        kafka_params,
        valueDecoder=lambda x: x.decode('utf-8')
    )
    
    return kafka_stream

def process_stream(kafka_stream):
    """
    Main stream processing logic with transformations and aggregations
    """
    # Extract log messages from Kafka (key, value) tuples
    log_lines = kafka_stream.map(lambda x: x[1])
    
    # Parse JSON log lines
    parsed_logs = log_lines.map(parse_log_line)
    
    # Filter for important logs (errors and warnings)
    important_logs = parsed_logs.filter(filter_error_logs)
    
    # Extract metrics for aggregation
    service_metrics = parsed_logs.map(extract_service_metrics)
    
    # Aggregate metrics per service within batch
    batch_aggregates = service_metrics.reduceByKey(aggregate_metrics)
    
    # Apply windowed aggregation for rolling statistics
    windowed_metrics = service_metrics.reduceByKeyAndWindow(
        aggregate_metrics,
        None,
        WINDOW_DURATION,
        SLIDE_INTERVAL
    )
    
    # Calculate final statistics
    service_stats = windowed_metrics.map(calculate_service_stats)
    
    # Save results to dashboard
    service_stats.foreachRDD(save_to_dashboard)
    
    # Print batch summary for monitoring
    batch_aggregates.foreachRDD(lambda rdd: 
        print(f"[{datetime.now()}] Processed batch with {rdd.count()} services") 
        if not rdd.isEmpty() else None
    )
    
    return service_stats

def graceful_shutdown(ssc):
    """
    Handle graceful shutdown on SIGTERM/SIGINT
    """
    def signal_handler(sig, frame):
        print("\n[INFO] Shutting down gracefully...")
        ssc.stop(stopSparkContext=True, stopGraceFully=True)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# ==================== APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    try:
        print(f"[INFO] Starting {APP_NAME} with batch interval: {batch_interval} seconds")
        print(f"[INFO] Kafka brokers: {KAFKA_BROKERS}")
        print(f"[INFO] Kafka topic: {KAFKA_TOPIC}")
        print(f"[INFO] Checkpoint directory: {CHECKPOINT_DIR}")
        
        # Create streaming context
        ssc = create_streaming_context()
        
        # Setup graceful shutdown handlers
        graceful_shutdown(ssc)
        
        # Create Kafka stream
        kafka_stream = setup_kafka_stream(ssc)
        
        # Process the stream
        process_stream(kafka_stream)
        
        # Start the streaming context
        print("[INFO] Starting streaming context...")
        ssc.start()
        
        # Wait for termination
        ssc.awaitTermination()
        
    except Exception as e:
        print(f"[ERROR] Application failed: {e}", file=sys.stderr)
        sys.exit(1)