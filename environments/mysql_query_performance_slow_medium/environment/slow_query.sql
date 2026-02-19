SELECT 
    orders.customer_id,
    COUNT(DISTINCT orders.id) as order_count,
    SUM(order_items.quantity) as total_items,
    SUM(order_items.quantity * order_items.unit_price) as total_spent
FROM orders
JOIN order_items ON orders.id = order_items.order_id
JOIN products ON order_items.product_id = products.id
WHERE orders.order_date >= '2023-06-01'
    AND orders.order_date < '2024-01-01'
    AND orders.status = 'completed'
    AND products.category = 'Electronics'
GROUP BY orders.customer_id
ORDER BY total_spent DESC;