-- Add email column to users table
-- This migration adds a unique email column for user authentication

ALTER TABLE users
ADD COLUMN email VARCHAR(255) NOT NULL UNIQUE;