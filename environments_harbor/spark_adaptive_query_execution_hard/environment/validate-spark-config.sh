#!/bin/bash

# Adaptive Query Execution Configuration Validator
# Validates Spark configuration files for AQE requirements

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config-file-path>"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "config_valid: no"
    echo "aqe_enabled: no"
    echo "validation_score: 0"
    exit 1
fi

# Initialize score
score=0
aqe_enabled="no"

# Function to get config value
get_config_value() {
    local key="$1"
    grep "^[^#]*${key}" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk -F'=' '{print $2}' | tr -d ' '
}

# Check 1: spark.sql.adaptive.enabled (20 points)
adaptive_enabled=$(get_config_value "spark.sql.adaptive.enabled")
if [ "$adaptive_enabled" = "true" ]; then
    score=$((score + 20))
    aqe_enabled="yes"
fi

# Check 2: spark.sql.adaptive.coalescePartitions.enabled (15 points)
coalesce_enabled=$(get_config_value "spark.sql.adaptive.coalescePartitions.enabled")
if [ "$coalesce_enabled" = "true" ]; then
    score=$((score + 15))
fi

# Check 3: spark.sql.adaptive.skewJoin.enabled (15 points)
skew_enabled=$(get_config_value "spark.sql.adaptive.skewJoin.enabled")
if [ "$skew_enabled" = "true" ]; then
    score=$((score + 15))
fi

# Check 4: spark.sql.autoBroadcastJoinThreshold (15 points)
# Should be between 10MB (10485760) and 100MB (104857600)
broadcast_threshold=$(get_config_value "spark.sql.autoBroadcastJoinThreshold")
if [ -n "$broadcast_threshold" ]; then
    if [ "$broadcast_threshold" -ge 10485760 ] && [ "$broadcast_threshold" -le 104857600 ]; then
        score=$((score + 15))
    fi
fi

# Check 5: spark.sql.shuffle.partitions (10 points)
# Should be between 100 and 400
shuffle_partitions=$(get_config_value "spark.sql.shuffle.partitions")
if [ -n "$shuffle_partitions" ]; then
    if [ "$shuffle_partitions" -ge 100 ] && [ "$shuffle_partitions" -le 400 ]; then
        score=$((score + 10))
    fi
fi

# Check 6: spark.sql.adaptive.advisoryPartitionSizeInBytes (15 points)
advisory_size=$(get_config_value "spark.sql.adaptive.advisoryPartitionSizeInBytes")
if [ -n "$advisory_size" ] && [ "$advisory_size" -gt 0 ]; then
    score=$((score + 15))
fi

# Check 7: spark.sql.adaptive.coalescePartitions.initialPartitionNum (10 points)
initial_partitions=$(get_config_value "spark.sql.adaptive.coalescePartitions.initialPartitionNum")
if [ -n "$initial_partitions" ] && [ "$initial_partitions" -gt 0 ]; then
    score=$((score + 10))
fi

# Determine if config is valid
config_valid="no"
if [ $score -ge 85 ]; then
    config_valid="yes"
fi

# Output results
echo "config_valid: $config_valid"
echo "aqe_enabled: $aqe_enabled"
echo "validation_score: $score"

exit 0