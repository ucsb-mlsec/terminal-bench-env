-- Transaction 1 (Order Processing)
-- This transaction processes a customer order by updating order status,
-- reserving items, and adjusting inventory levels.

START TRANSACTION;

-- Locks orders table first
UPDATE orders SET status = 'processing' WHERE order_id = 1001;

-- Then locks order_items table
UPDATE order_items SET status = 'reserved' WHERE order_id = 1001;

-- Finally locks inventory table
UPDATE inventory SET quantity_available = quantity_available - 5 WHERE product_id = 101;

COMMIT;


-- Transaction 2 (Inventory Update)
-- This transaction updates inventory reservations and verifies related orders
-- based on inventory availability checks.

START TRANSACTION;

-- Locks inventory table first
UPDATE inventory SET reserved_quantity = reserved_quantity + 3 WHERE product_id = 101;

-- Then reads from order_items table (acquires shared lock)
SELECT order_id FROM order_items WHERE product_id = 101 AND status = 'pending';

-- Finally locks orders table (needs exclusive lock)
UPDATE orders SET status = 'verified' WHERE order_id = 1001;

COMMIT;


-- LOCK ACQUISITION SUMMARY:
-- Transaction 1: orders -> order_items -> inventory
-- Transaction 2: inventory -> order_items -> orders
-- These opposite orderings create a circular wait condition leading to deadlocks