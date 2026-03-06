#include "../include/processing.h"
#include "../include/utils.h"
#include <string.h>

// Static buffer for intermediate processing
static int intermediate_buffer[MAX_BUFFER_SIZE];
static double filter_coefficients[FILTER_ORDER] = {0.1, 0.2, 0.4, 0.2, 0.1};

// Process raw sensor data and apply scaling
int process_data(const int* input, int* output, int length) {
    if (!input || !output || length <= 0 || length > MAX_BUFFER_SIZE) {
        return -1;
    }

    for (int i = 0; i < length; i++) {
        // Apply scaling and offset
        int scaled = (input[i] * SCALE_FACTOR) / 100;
        output[i] = scaled + OFFSET_VALUE;
        
        // Clamp values to valid range
        if (output[i] < MIN_VALUE) {
            output[i] = MIN_VALUE;
        } else if (output[i] > MAX_VALUE) {
            output[i] = MAX_VALUE;
        }
    }

    return 0;
}

// Transform input data using lookup table approach
int transform_input(int* data, int length) {
    if (!data || length <= 0 || length > MAX_BUFFER_SIZE) {
        return -1;
    }

    // Copy to intermediate buffer
    memcpy(intermediate_buffer, data, length * sizeof(int));

    // Apply transformation
    for (int i = 0; i < length; i++) {
        int value = intermediate_buffer[i];
        
        // Non-linear transformation for sensor calibration
        if (value < 0) {
            data[i] = -(value * value) / 1000;
        } else if (value < 500) {
            data[i] = value * 2;
        } else if (value < 1000) {
            data[i] = 1000 + (value - 500) / 2;
        } else {
            data[i] = 1250;
        }
    }

    return 0;
}

// Compute statistical result from data array
int compute_result(const int* data, int length, struct result_data* result) {
    if (!data || !result || length <= 0 || length > MAX_BUFFER_SIZE) {
        return -1;
    }

    long long sum = 0;
    int min = data[0];
    int max = data[0];

    // Calculate statistics
    for (int i = 0; i < length; i++) {
        sum += data[i];
        
        if (data[i] < min) {
            min = data[i];
        }
        if (data[i] > max) {
            max = data[i];
        }
    }

    result->average = (int)(sum / length);
    result->min = min;
    result->max = max;
    result->range = max - min;

    // Calculate variance
    long long variance_sum = 0;
    for (int i = 0; i < length; i++) {
        int diff = data[i] - result->average;
        variance_sum += (long long)diff * diff;
    }
    result->variance = (int)(variance_sum / length);

    return 0;
}

// Apply FIR filter to data stream
int apply_filter(const int* input, int* output, int length) {
    if (!input || !output || length <= 0 || length > MAX_BUFFER_SIZE) {
        return -1;
    }

    // Apply FIR filter with specified coefficients
    for (int i = 0; i < length; i++) {
        double sum = 0.0;
        
        for (int j = 0; j < FILTER_ORDER; j++) {
            int idx = i - j;
            if (idx >= 0) {
                sum += input[idx] * filter_coefficients[j];
            } else {
                sum += 0; // Zero padding for edges
            }
        }
        
        output[i] = (int)sum;
    }

    return 0;
}

// Detect peaks in signal data
int detect_peaks(const int* data, int length, int* peak_indices, int max_peaks) {
    if (!data || !peak_indices || length < 3 || max_peaks <= 0) {
        return -1;
    }

    int peak_count = 0;
    int threshold = compute_threshold(data, length);

    // Find local maxima above threshold
    for (int i = 1; i < length - 1 && peak_count < max_peaks; i++) {
        if (data[i] > threshold && 
            data[i] > data[i-1] && 
            data[i] > data[i+1]) {
            peak_indices[peak_count++] = i;
        }
    }

    return peak_count;
}

// Smooth data using moving average
void smooth_data(int* data, int length, int window_size) {
    if (!data || length <= 0 || window_size <= 0 || window_size > length) {
        return;
    }

    memcpy(intermediate_buffer, data, length * sizeof(int));

    for (int i = 0; i < length; i++) {
        int sum = 0;
        int count = 0;

        int start = (i - window_size / 2 < 0) ? 0 : i - window_size / 2;
        int end = (i + window_size / 2 >= length) ? length - 1 : i + window_size / 2;

        for (int j = start; j <= end; j++) {
            sum += intermediate_buffer[j];
            count++;
        }

        data[i] = sum / count;
    }
}