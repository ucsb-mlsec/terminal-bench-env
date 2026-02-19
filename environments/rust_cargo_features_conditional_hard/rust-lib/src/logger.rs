// This file contains logging functionality
// Should only be available when the 'logging' feature is enabled

pub struct Logger {
    enabled: bool,
}

impl Logger {
    pub fn new() -> Self {
        Logger { enabled: true }
    }

    pub fn log(&self, message: &str) {
        if self.enabled {
            println!("[LOG] {}", message);
        }
    }
}

pub fn init_logger() {
    println!("Logger initialized");
}

#[cfg(feature = "logging")]
fn internal_helper() {
    // Internal helper function
}

// Some utility that might be incorrectly gated
pub fn get_log_level() -> u8 {
    1
}