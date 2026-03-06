CREATE MATERIALIZED VIEW aggregated_metrics_mv TO aggregated_metrics_dest AS
SELECT 
    hour,
    sum(event_count) as total_events,
    count(distinct event_type) as unique_types
FROM events_hourly_dest
GROUP BY hour;