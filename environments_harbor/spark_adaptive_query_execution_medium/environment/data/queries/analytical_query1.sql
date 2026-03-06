SELECT 
    c.customer_id,
    c.customer_name,
    c.region,
    p.product_category,
    COUNT(DISTINCT s.order_id) as total_orders,
    SUM(s.order_amount) as total_revenue,
    AVG(s.order_amount) as avg_order_value,
    ROW_NUMBER() OVER (PARTITION BY c.region ORDER BY SUM(s.order_amount) DESC) as revenue_rank
FROM 
    sales_data s
    INNER JOIN customer_info c ON s.customer_id = c.customer_id
    INNER JOIN product_catalog p ON s.product_id = p.product_id
WHERE 
    s.order_date >= '2023-01-01'
    AND s.order_date < '2024-01-01'
    AND s.order_status = 'completed'
    AND c.customer_status = 'active'
GROUP BY 
    c.customer_id, c.customer_name, c.region, p.product_category
HAVING 
    SUM(s.order_amount) > 1000
ORDER BY 
    total_revenue DESC