CREATE MATERIALIZED VIEW daily_summary_mv TO daily_summary_dest AS
SELECT 
    toDate(timestamp) as date,
    count() as daily_count,
    uniq(user_id) as unique_users
FROM raw_events
GROUP BY date;