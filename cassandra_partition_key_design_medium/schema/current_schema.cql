-- Current problematic schema - partitions by date only
-- This causes all sensor data for a given day to write to a single partition
-- Results in partitions exceeding 2GB and severe hotspots on individual nodes

CREATE TABLE sensor_readings (
    date text,                    -- Partition key: only by date (PROBLEM - causes massive partitions)
    timestamp timestamp,          -- Clustering key: orders data within partition
    sensor_id text,               -- Clustering key: secondary sort
    temperature decimal,
    location text,
    device_type text,
    PRIMARY KEY (date, timestamp, sensor_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, sensor_id ASC);