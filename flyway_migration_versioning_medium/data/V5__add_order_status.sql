-- Migration: Add order status column
-- Version: V5
-- Description: Adds a status column to the orders table to track order state
-- Created: 2024-01-15

-- Add status column to orders table
ALTER TABLE orders
ADD COLUMN status VARCHAR(50) NOT NULL DEFAULT 'pending';

-- Add check constraint to ensure valid status values
ALTER TABLE orders
ADD CONSTRAINT chk_order_status 
CHECK (status IN ('pending', 'completed', 'cancelled', 'processing', 'shipped'));

-- Create index on status for faster queries
CREATE INDEX idx_orders_status ON orders(status);

-- Update existing orders to have a default status
UPDATE orders 
SET status = 'completed' 
WHERE created_at < NOW() - INTERVAL '30 days';