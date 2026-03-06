-- User Events Table
-- Stores user activity events with flexible JSONB data structure

CREATE TABLE user_events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    event_data JSONB NOT NULL
);

COMMENT ON TABLE user_events IS 'Stores user activity events from microservices application';
COMMENT ON COLUMN user_events.event_data IS 'JSONB column containing flexible event properties including user_id, event_type, tags, metadata, and other contextual information';