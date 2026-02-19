use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Represents a user in the system
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Generic API response wrapper
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: String,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: String::from("Operation completed successfully"),
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            message,
        }
    }
}

/// Represents an event in the system
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Event {
    pub id: u64,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub payload: String,
}