SELECT
    e.user_id,
    u.username,
    u.account_type,
    p.product_id,
    p.product_name,
    p.product_category,
    COUNT(*) as event_count,
    SUM(e.duration_seconds) as total_time_spent,
    AVG(e.duration_seconds) as avg_session_duration,
    COUNT(DISTINCT e.session_id) as unique_sessions,
    SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) as purchase_count,
    AVG(e.engagement_score) as avg_engagement_score
FROM user_events e
INNER JOIN user_profiles u ON e.user_id = u.user_id
INNER JOIN products p ON e.product_id = p.product_id
WHERE e.event_date >= '2024-01-01'
    AND e.event_date < '2024-12-31'
    AND u.is_active = true
    AND e.event_type IN ('view', 'click', 'purchase', 'add_to_cart')
GROUP BY 
    e.user_id,
    u.username,
    u.account_type,
    p.product_id,
    p.product_name,
    p.product_category
ORDER BY event_count DESC, total_time_spent DESC
LIMIT 10000