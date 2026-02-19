mod core;

#[cfg(feature = "database")]
mod database;

#[cfg(feature = "analytics")]
mod analytics;

fn main() {
    println!("Starting data processor...");
    
    core::process_basic_data();
    
    #[cfg(feature = "database")]
    database::init_db();
    
    #[cfg(feature = "analytics")]
    analytics::run_analytics();
    
    println!("Processing complete");
}