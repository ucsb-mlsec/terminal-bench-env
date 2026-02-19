-- This query joins customer and transaction tables to generate aggregated sales metrics
-- PERFORMANCE ISSUE: Experiencing severe performance degradation due to data skew
-- Problem: Certain customer IDs have disproportionately high transaction volumes
-- Impact: Some partitions are processing 100x more data than others, causing timeouts
-- The skewed partitions are creating bottlenecks that delay the entire job completion
-- Solution needed: Enable Spark's adaptive execution and skewed join optimization

SELECT 
    c.customer_id,
    c.customer_name,
    c.region,
    c.account_type,
    COUNT(t.transaction_id) as transaction_count,
    SUM(t.amount) as total_amount,
    AVG(t.amount) as avg_amount
FROM customers c
INNER JOIN transactions t 
    ON c.customer_id = t.customer_id
GROUP BY 
    c.customer_id, 
    c.customer_name, 
    c.region, 
    c.account_type
ORDER BY transaction_count DESC;