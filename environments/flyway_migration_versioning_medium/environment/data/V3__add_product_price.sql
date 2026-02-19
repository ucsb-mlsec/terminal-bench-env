-- Add price column to products table
-- This migration adds a price field to track product pricing
-- Created: 2024-01-15

ALTER TABLE products ADD COLUMN price DECIMAL(10, 2) NOT NULL DEFAULT 0.00;