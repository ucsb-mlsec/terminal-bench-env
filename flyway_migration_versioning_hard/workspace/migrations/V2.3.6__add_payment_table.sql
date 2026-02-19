-- Flyway migration: Add payment table
-- Version: 2.3.6
-- Description: Create payments table to track order payments

CREATE TABLE payments (
    id INT PRIMARY KEY,
    order_id INT,
    amount DECIMAL(10,2),
    payment_date TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);