SELECT 
    u.user_id,
    u.username,
    u.country,
    COUNT(t.transaction_id) as total_transactions,
    SUM(t.amount) as total_spent,
    AVG(t.amount) as avg_transaction_amount
FROM transactions t
JOIN users u ON t.user_id = u.user_id
WHERE t.transaction_date >= '2024-01-01'
    AND t.transaction_date < '2024-12-31'
    AND t.status = 'completed'
GROUP BY u.user_id, u.username, u.country
HAVING COUNT(t.transaction_id) > 5
ORDER BY total_spent DESC
LIMIT 100