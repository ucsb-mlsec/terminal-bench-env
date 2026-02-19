-- Add created_at timestamp column to users table
-- This migration adds tracking for when user records are created

ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL;