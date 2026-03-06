CREATE MATERIALIZED VIEW user_summary_mv TO user_summary_dest AS
SELECT 
    user_id,
    count() as action_count,
    avg(duration) as avg_duration
FROM user_events
GROUP BY user_id;