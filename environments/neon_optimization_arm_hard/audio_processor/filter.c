#include <stdint.h>
#include <string.h>

// Compute-intensive filter operations that would benefit from SIMD optimization
// These functions process audio samples with mathematical operations suitable for vectorization

// Global state variables for filter coefficients
static float prev_sample_low = 0.0f;
static float prev_sample_high = 0.0f;

// Low-pass filter implementation
// Processes audio samples to attenuate high frequencies
// This loop-based approach is ideal for NEON vectorization
void apply_lowpass_filter(int16_t* samples, int count, float cutoff) {
    float alpha = cutoff / (cutoff + 1.0f);
    float beta = 1.0f - alpha;
    
    // Compute-intensive loop - processes each sample sequentially
    // Array operations here could be vectorized with SIMD instructions
    for (int i = 0; i < count; i++) {
        // Convert to float for processing
        float current = (float)samples[i];
        
        // Apply filter coefficients - multiplication and addition operations
        float filtered = (alpha * current) + (beta * prev_sample_low);
        
        // Apply gain scaling
        filtered = filtered * 0.95f;
        
        // Store for next iteration
        prev_sample_low = filtered;
        
        // Convert back to int16_t with saturation
        if (filtered > 32767.0f) filtered = 32767.0f;
        if (filtered < -32768.0f) filtered = -32768.0f;
        
        samples[i] = (int16_t)filtered;
    }
}

// High-pass filter implementation
// Processes audio samples to attenuate low frequencies
// Similar structure to low-pass, suitable for NEON optimization
void apply_highpass_filter(int16_t* samples, int count, float cutoff) {
    float alpha = 1.0f / (cutoff + 1.0f);
    float gain = 1.2f;
    
    // Compute-intensive loop - processes samples with arithmetic operations
    // These array operations are prime candidates for vectorization
    for (int i = 0; i < count; i++) {
        // Convert sample to floating point
        float current = (float)samples[i];
        
        // High-pass filter calculation - subtraction and multiplication
        float filtered = alpha * (prev_sample_high + current - prev_sample_low);
        
        // Apply gain compensation
        filtered = filtered * gain;
        
        // Update state
        prev_sample_low = current;
        prev_sample_high = filtered;
        
        // Clamp to int16_t range and convert back
        if (filtered > 32767.0f) filtered = 32767.0f;
        if (filtered < -32768.0f) filtered = -32768.0f;
        
        samples[i] = (int16_t)filtered;
    }
}