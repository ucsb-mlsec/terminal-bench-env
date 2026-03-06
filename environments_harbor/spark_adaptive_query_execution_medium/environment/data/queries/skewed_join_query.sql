-- Sample query demonstrating data skew in join operations
-- Common scenario: Popular users dominate the distribution causing skewed partitions

SELECT 
    u.user_id,
    u.user_type,
    COUNT(DISTINCT s.session_id) as session_count,
    COUNT(c.click_id) as total_clicks,
    SUM(c.click_value) as total_value,
    AVG(s.session_duration) as avg_session_duration
FROM user_events u
-- This join on user_id often has skewed distribution (power users with 1000x more events)
INNER JOIN sessions s ON u.user_id = s.user_id
-- Click stream join compounds the skew problem
INNER JOIN click_stream c ON s.session_id = c.session_id
WHERE u.event_date >= '2024-01-01'
    AND u.user_type IN ('premium', 'enterprise')
    AND s.session_duration > 60
-- GROUP BY on skewed key causes uneven partition sizes
GROUP BY u.user_id, u.user_type
HAVING COUNT(DISTINCT s.session_id) > 5
ORDER BY total_value DESC
LIMIT 1000;