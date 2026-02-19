// Library for demonstrating conditional compilation issues
#![warn(missing_docs)]

//! A utility library with optional features for logging, networking, and compression.

// Module declarations with incorrect/missing feature guards
pub mod logger;

#[cfg(feature = "network")]  // WRONG: should be "networking"
pub mod network;

pub mod compress;  // MISSING: should have cfg guard for "compression" feature

/// Initialize the library
pub fn init() {
    // Calls without proper feature guards
    logger::setup();
    network::connect();  // MISSING: should be feature-gated
    compress::init();     // MISSING: should be feature-gated
}

/// Get library version
pub fn version() -> &'static str {
    "1.0.0"
}

#[cfg(feature = "logging")]
/// Configure logging level
pub fn set_log_level(level: u8) {
    logger::set_level(level);
}

// MISSING: networking feature should require logging feature check
#[cfg(feature = "networking")]
/// Start network service
pub fn start_network() {
    logger::log("Starting network");  // MISSING: guard for logger
    network::start();
}

#[cfg(feature = "compress")]  // WRONG: should be "compression"
/// Compress data
pub fn compress_data(data: &[u8]) -> Vec<u8> {
    compress::compress(data)
}

/// Utility function available always
pub fn utility_function() -> i32 {
    42
}