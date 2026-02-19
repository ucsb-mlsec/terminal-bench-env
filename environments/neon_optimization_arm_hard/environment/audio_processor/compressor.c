#include <stdint.h>
#include <math.h>
#include <stdlib.h>

/* Performance-critical audio processing */
/* Dynamic range compressor for real-time audio */

#define MIN_SAMPLE -32768
#define MAX_SAMPLE 32767

/* Apply dynamic range compression to audio samples */
void apply_compression(int16_t* samples, int count, float threshold, float ratio) {
    if (samples == NULL || count <= 0) {
        return;
    }
    
    /* Process each sample individually */
    for (int i = 0; i < count; i++) {
        float sample = (float)samples[i];
        float abs_sample = fabsf(sample);
        
        /* Check if sample exceeds threshold */
        if (abs_sample > threshold) {
            /* Calculate gain reduction */
            float excess = abs_sample - threshold;
            float compressed = excess / ratio;
            float new_level = threshold + compressed;
            
            /* Apply gain reduction maintaining sign */
            float gain = new_level / abs_sample;
            sample = sample * gain;
            
            /* Clamp to valid range */
            if (sample > MAX_SAMPLE) {
                sample = MAX_SAMPLE;
            } else if (sample < MIN_SAMPLE) {
                sample = MIN_SAMPLE;
            }
            
            samples[i] = (int16_t)sample;
        }
    }
}

/* Apply soft knee compression for smoother transitions */
void apply_soft_knee(int16_t* samples, int count, float knee_width) {
    if (samples == NULL || count <= 0 || knee_width <= 0.0f) {
        return;
    }
    
    float knee_start = 0.5f - (knee_width / 2.0f);
    float knee_end = 0.5f + (knee_width / 2.0f);
    
    /* Performance-critical audio processing loop */
    for (int i = 0; i < count; i++) {
        float sample = (float)samples[i];
        float normalized = fabsf(sample) / 32768.0f;
        
        /* Calculate soft knee gain curve */
        if (normalized > knee_start && normalized < knee_end) {
            float position = (normalized - knee_start) / knee_width;
            float gain = 1.0f - (position * position * 0.3f);
            sample = sample * gain;
            
            /* Clamp output */
            if (sample > MAX_SAMPLE) {
                sample = MAX_SAMPLE;
            } else if (sample < MIN_SAMPLE) {
                sample = MIN_SAMPLE;
            }
            
            samples[i] = (int16_t)sample;
        }
    }
}

/* Calculate RMS level for compression metering */
float calculate_rms(int16_t* samples, int count) {
    if (samples == NULL || count <= 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float sample = (float)samples[i] / 32768.0f;
        sum += sample * sample;
    }
    
    return sqrtf(sum / (float)count);
}