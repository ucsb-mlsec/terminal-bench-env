CREATE MATERIALIZED VIEW orphaned_view_mv TO orphaned_dest AS
SELECT 
    user_id,
    sum(metric_value) as total_value
FROM missing_view_dest
GROUP BY user_id;