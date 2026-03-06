-- Flyway migration V2.3.0
-- Description: Create audit log table for tracking database changes
-- This migration creates the audit_log table for compliance requirements

CREATE TABLE audit_log (
    id INT PRIMARY KEY,
    table_name VARCHAR(50),
    action VARCHAR(20),
    timestamp TIMESTAMP
);