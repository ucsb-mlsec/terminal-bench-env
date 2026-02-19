#include "signal.h"
#include <stdlib.h>

void low_pass_filter(double *signal, int length, double alpha) {
    if (signal == NULL || length <= 0 || alpha < 0.0 || alpha > 1.0) {
        return;
    }
    
    double prev = signal[0];
    for (int i = 1; i < length; i++) {
        signal[i] = alpha * signal[i] + (1.0 - alpha) * prev;
        prev = signal[i];
    }
}

void high_pass_filter(double *signal, int length, double alpha) {
    if (signal == NULL || length <= 0 || alpha < 0.0 || alpha > 1.0) {
        return;
    }
    
    double *temp = (double *)malloc(length * sizeof(double));
    if (temp == NULL) {
        return;
    }
    
    temp[0] = signal[0];
    for (int i = 0; i < length; i++) {
        temp[i] = signal[i];
    }
    
    double prev_input = temp[0];
    double prev_output = temp[0];
    
    for (int i = 1; i < length; i++) {
        signal[i] = alpha * (prev_output + temp[i] - prev_input);
        prev_input = temp[i];
        prev_output = signal[i];
    }
    
    free(temp);
}