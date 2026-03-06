CREATE MATERIALIZED VIEW events_hourly_mv TO events_hourly_dest AS
SELECT 
    toStartOfHour(timestamp) as hour,
    event_type,
    count() as event_count
FROM raw_events
GROUP BY hour, event_type;