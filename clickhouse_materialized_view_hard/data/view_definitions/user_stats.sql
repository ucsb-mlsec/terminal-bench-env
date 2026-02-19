CREATE MATERIALIZED VIEW user_stats_mv TO user_stats_dest AS
SELECT 
    user_id,
    count() as total_actions,
    max(timestamp) as last_seen
FROM user_events
GROUP BY user_id;