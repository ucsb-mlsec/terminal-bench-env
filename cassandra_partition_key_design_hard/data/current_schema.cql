-- Current Production Schema (PROBLEMATIC - DO NOT USE)
-- This schema has critical performance issues under production load

CREATE TABLE sensor_readings (
    sensor_id text,
    timestamp timestamp,
    data_center text,
    reading_type text,
    value double,
    PRIMARY KEY (data_center, timestamp, sensor_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, sensor_id ASC)
  AND compaction = {'class': 'SizeTieredCompactionStrategy'}
  AND gc_grace_seconds = 864000;

-- KNOWN ISSUES WITH THIS SCHEMA:
-- 1. HOT PARTITIONS: Using data_center as partition key causes all 15,000+ sensors
--    in a data center to write to the same partition, creating massive write hotspots
--    under 50,000 writes/second peak load
--
-- 2. UNBOUNDED GROWTH: Each data_center partition grows indefinitely as time passes.
--    With 15,000 sensors * 120 readings/hour * 24 hours * 90 days = 388+ million rows
--    per data center partition, causing severe read/write performance degradation
--
-- 3. POOR DISTRIBUTION: Only 3-5 data centers means only 3-5 partitions are being
--    used across potentially hundreds of Cassandra nodes, leaving most nodes idle
--
-- 4. MEMORY PRESSURE: Massive partitions cause heap pressure, compaction problems,
--    and eventual node failures