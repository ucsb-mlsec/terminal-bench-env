-- Flyway migration V2.3.5
-- Description: Modify orders table to add status tracking
-- Author: Development Team B
-- Date: 2024-01-15

ALTER TABLE orders ADD COLUMN status VARCHAR(20) DEFAULT 'pending';