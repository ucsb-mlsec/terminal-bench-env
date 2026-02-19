-- Flyway migration: Add shipping table
-- Version: 2.4.0
-- Description: Create shipping table to track order shipments

CREATE TABLE shipping (
    id INT PRIMARY KEY,
    order_id INT,
    address VARCHAR(200),
    shipped_date TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

-- Create index on order_id for faster lookups
CREATE INDEX idx_shipping_order_id ON shipping(order_id);

-- Create index on shipped_date for reporting queries
CREATE INDEX idx_shipping_shipped_date ON shipping(shipped_date);