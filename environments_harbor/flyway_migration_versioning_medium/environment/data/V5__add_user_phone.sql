-- Add phone column to users table for contact information
-- This migration adds a phone number field to store user contact details

ALTER TABLE users ADD COLUMN phone VARCHAR(20);