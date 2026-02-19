-- Query 1: Filter by scalar value - get events for specific user
-- Uses ->> operator to extract text value
SELECT id, event_data, created_at
FROM user_events
WHERE event_data->>'user_id' = '12345';

-- Query 2: Containment check - find events with specific properties
-- Uses @> operator to check if JSONB contains key-value pairs
SELECT id, event_data, created_at
FROM user_events
WHERE event_data @> '{"event_type": "purchase", "status": "completed"}';

-- Query 3: Key existence check - find events that have a specific key
-- Uses ? operator to check for top-level key existence
SELECT id, event_data, created_at
FROM user_events
WHERE event_data ? 'transaction_id';

-- Query 4: Nested property access - filter by nested value
-- Uses -> followed by ->> to access nested JSONB properties
SELECT id, event_data, created_at
FROM user_events
WHERE event_data->'metadata'->>'source' = 'mobile_app';

-- Query 5: Multiple key existence check - events with any of several keys
-- Uses ?| operator to check if any of the specified keys exist
SELECT id, event_data, created_at
FROM user_events
WHERE event_data ?| ARRAY['error_code', 'failure_reason', 'exception'];

-- Query 6: Complex filter with multiple JSONB conditions
-- Combines multiple JSONB operations with AND/OR
SELECT id, event_data, created_at
FROM user_events
WHERE event_data->>'event_type' = 'login'
  AND event_data @> '{"successful": true}'
  AND event_data->'details'->>'ip_address' IS NOT NULL;