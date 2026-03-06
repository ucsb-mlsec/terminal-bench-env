-- Add category column to products table
-- This migration adds support for product categorization

ALTER TABLE products
ADD COLUMN category VARCHAR(100);