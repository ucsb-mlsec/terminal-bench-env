use std::fs;

fn debug_log(message: &str) {
    let _ = fs::write("/tmp/calculator_debug.log", message);
}

#[no_mangle]
pub extern "C" fn add(a: f64, b: f64) -> f64 {
    debug_log(&format!("Called add with {} and {}", a, b));
    a + b
}

#[no_mangle]
pub extern "C" fn subtract(a: f64, b: f64) -> f64 {
    debug_log(&format!("Called subtract with {} and {}", a, b));
    a - b
}

#[no_mangle]
pub extern "C" fn multiply(a: f64, b: f64) -> f64 {
    debug_log(&format!("Called multiply with {} and {}", a, b));
    a * b
}

#[no_mangle]
pub extern "C" fn divide(a: f64, b: f64) -> f64 {
    debug_log(&format!("Called divide with {} and {}", a, b));
    a / b
}