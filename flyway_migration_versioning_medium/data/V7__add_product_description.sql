-- Add description column to products table
-- This migration adds a TEXT column to store detailed product descriptions

ALTER TABLE products ADD COLUMN description TEXT;