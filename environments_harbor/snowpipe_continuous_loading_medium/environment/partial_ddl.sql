-- Database and Schema Setup
CREATE DATABASE IF NOT EXISTS ANALYTICS_DB;
-- TODO: Add schema creation

-- Table Definition
CREATE OR REPLACE TABLE ANALYTICS_DB.STAGING.TRANSACTIONS (
    transaction_id VARCHAR(50),
    customer_id VARCHAR(50)
    -- TODO: Add remaining columns
    -- Expected columns based on sample data:
    -- transaction_date, amount, payment_method, status, merchant_id, etc.
);

-- Stage Creation
CREATE OR REPLACE STAGE ANALYTICS_DB.STAGING.S3_TRANSACTION_STAGE
    URL = 's3://company-data-lake/transactions/incoming/'
    -- TODO: Add storage integration and credentials
    -- Need to configure:
    -- - STORAGE_INTEGRATION or CREDENTIALS
    -- - FILE_FORMAT reference
;

-- TODO: Create file format for pipe-delimited CSV with header
-- Requirements:
-- - Pipe delimiter (|)
-- - Skip header row
-- - Handle null values
-- - Timestamp parsing for YYYY-MM-DD HH24:MI:SS format
-- - Error handling for data quality issues

-- TODO: Create Snowpipe with auto-ingest
-- Requirements:
-- - Enable AUTO_INGEST for S3 event notifications
-- - Reference the stage and file format
-- - Configure to process files within 1 minute
-- - Handle files up to 100MB
-- - Error handling to skip bad rows but log them