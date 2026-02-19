//! Data processing library with optional database and analytics features

pub mod core;

#[cfg(feature = "database")]
pub mod database;

#[cfg(feature = "analytics")]
pub mod analytics;