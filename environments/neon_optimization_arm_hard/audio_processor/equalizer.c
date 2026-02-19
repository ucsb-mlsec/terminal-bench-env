#include <stdint.h>
#include <math.h>

#define NUM_BANDS 10
#define SAMPLE_RATE 48000
#define PI 3.14159265358979323846

// Structure to hold filter state for each band
typedef struct {
    float b0, b1, b2, a1, a2;
    float x1, x2, y1, y2;
} BiquadFilter;

static BiquadFilter band_filters[NUM_BANDS];

// Helper function to calculate band coefficients using trigonometric operations
// These are vectorizable operations that benefit from SIMD
void calculate_band_coefficients(float* coeffs, int band) {
    float center_freq = 60.0f * powf(2.0f, band * 1.2f);
    float omega = 2.0f * PI * center_freq / SAMPLE_RATE;
    float q_factor = 1.4142f;
    
    // DSP processing: biquad filter coefficient calculation
    float alpha = sinf(omega) / (2.0f * q_factor);
    float cos_omega = cosf(omega);
    
    coeffs[0] = 1.0f + alpha;  // a0
    coeffs[1] = -2.0f * cos_omega;  // a1
    coeffs[2] = 1.0f - alpha;  // a2
    coeffs[3] = (1.0f + cos_omega) / 2.0f;  // b0
    coeffs[4] = -(1.0f + cos_omega);  // b1
    coeffs[5] = (1.0f + cos_omega) / 2.0f;  // b2
}

// Initialize equalizer band filters
void init_equalizer() {
    float coeffs[6];
    for (int band = 0; band < NUM_BANDS; band++) {
        calculate_band_coefficients(coeffs, band);
        
        band_filters[band].b0 = coeffs[3] / coeffs[0];
        band_filters[band].b1 = coeffs[4] / coeffs[0];
        band_filters[band].b2 = coeffs[5] / coeffs[0];
        band_filters[band].a1 = coeffs[1] / coeffs[0];
        band_filters[band].a2 = coeffs[2] / coeffs[0];
        band_filters[band].x1 = 0.0f;
        band_filters[band].x2 = 0.0f;
        band_filters[band].y1 = 0.0f;
        band_filters[band].y2 = 0.0f;
    }
}

// Main equalizer function with vectorizable operations
// Applies gain adjustments across multiple frequency bands
void apply_equalizer(int16_t* samples, int count, float* gains) {
    float temp_buffer[count];
    
    // Convert samples to float for DSP processing
    for (int i = 0; i < count; i++) {
        temp_buffer[i] = (float)samples[i];
    }
    
    // Process each frequency band - vectorizable operations
    for (int band = 0; band < NUM_BANDS; band++) {
        BiquadFilter* filter = &band_filters[band];
        float gain = gains[band];
        
        // Inner loop for processing samples with biquad filter
        // These floating-point multiplications and accumulations are ideal for NEON
        for (int i = 0; i < count; i++) {
            float input = temp_buffer[i];
            
            // Biquad difference equation - DSP processing core
            float output = filter->b0 * input + 
                          filter->b1 * filter->x1 + 
                          filter->b2 * filter->x2 -
                          filter->a1 * filter->y1 - 
                          filter->a2 * filter->y2;
            
            // Update filter state
            filter->x2 = filter->x1;
            filter->x1 = input;
            filter->y2 = filter->y1;
            filter->y1 = output;
            
            // Apply band gain and accumulate
            temp_buffer[i] += output * gain;
        }
    }
    
    // Convert back to int16_t with clipping
    for (int i = 0; i < count; i++) {
        float sample = temp_buffer[i];
        if (sample > 32767.0f) sample = 32767.0f;
        if (sample < -32768.0f) sample = -32768.0f;
        samples[i] = (int16_t)sample;
    }
}