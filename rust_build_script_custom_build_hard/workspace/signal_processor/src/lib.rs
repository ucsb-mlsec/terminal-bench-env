// Signal processing library with C FFI bindings

use std::os::raw::c_int;

// External C function declarations
extern "C" {
    /// Applies a low-pass filter to the signal data
    /// 
    /// # Parameters
    /// - `data`: pointer to array of samples
    /// - `length`: number of samples in the array
    /// - `cutoff_freq`: cutoff frequency for the filter
    /// - `sample_rate`: sampling rate of the signal
    fn low_pass_filter(
        data: *mut f64,
        length: c_int,
        cutoff_freq: f64,
        sample_rate: f64,
    ) -> c_int;

    /// Calculates the RMS (Root Mean Square) value of a signal
    /// 
    /// # Parameters
    /// - `data`: pointer to array of samples
    /// - `length`: number of samples in the array
    fn calculate_rms(data: *const f64, length: c_int) -> f64;

    /// Normalizes signal data to a specified peak amplitude
    /// 
    /// # Parameters
    /// - `data`: pointer to array of samples
    /// - `length`: number of samples in the array
    /// - `target_peak`: desired peak amplitude
    fn normalize_signal(data: *mut f64, length: c_int, target_peak: f64);
}

// Safe Rust wrapper for RMS calculation
pub fn compute_rms(samples: &[f64]) -> f64 {
    unsafe {
        calculate_rms(samples.as_ptr(), samples.len() as c_int)
    }
}

// Safe Rust wrapper for low-pass filter
pub fn apply_lowpass(samples: &mut [f64], cutoff: f64, sample_rate: f64) -> Result<(), String> {
    let result = unsafe {
        low_pass_filter(
            samples.as_mut_ptr(),
            samples.len() as c_int,
            cutoff,
            sample_rate,
        )
    };
    
    if result == 0 {
        Ok(())
    } else {
        Err(format!("Low-pass filter failed with code: {}", result))
    }
}

// Safe Rust wrapper for signal normalization
pub fn normalize(samples: &mut [f64], target_peak: f64) {
    unsafe {
        normalize_signal(samples.as_mut_ptr(), samples.len() as c_int, target_peak);
    }
}