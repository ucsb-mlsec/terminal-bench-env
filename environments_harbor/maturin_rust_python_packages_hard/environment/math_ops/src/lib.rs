use pyo3::prelude::*;

/// Add two numbers together
#[pyfunction]
fn add(a: f64, b: f64) -> PyResult<f64> {
    Ok(a + b)
}

/// Multiply two numbers
#[pyfunction]
fn multiply(a: f64, b: f64) -> PyResult<f64> {
    Ok(a * b)
}

/// Raise a number to a power
#[pyfunction]
fn power(base: f64, exponent: f64) -> PyResult<f64> {
    Ok(base.powf(exponent))
}

/// Calculate the factorial of a non-negative integer
#[pyfunction]
fn factorial(n: u64) -> PyResult<u64> {
    if n == 0 {
        return Ok(1);
    }
    let mut result: u64 = 1;
    for i in 1..=n {
        result = result.checked_mul(i)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyOverflowError, _>("Factorial overflow"))?;
    }
    Ok(result)
}

/// A Python module implemented in Rust for mathematical operations
#[pymodule]
fn math_ops(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(power, m)?)?;
    m.add_function(wrap_pyfunction!(factorial, m)?)?;
    Ok(())
}