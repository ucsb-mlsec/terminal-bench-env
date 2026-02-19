#include <stdio.h>
#include <stdlib.h>

double* add_signals(double* sig1, double* sig2, int length) {
    double* result = (double*)malloc(length * sizeof(double));
    if (result == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < length; i++) {
        result[i] = sig1[i] + sig2[i];
    }
    
    return result;
}

double* multiply_signal(double* signal, int length, double multiplier) {
    double* result = (double*)malloc(length * sizeof(double));
    if (result == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < length; i++) {
        result[i] = signal[i] * multiplier;
    }
    
    return result;
}

double mean_signal(double* signal, int length) {
    if (length <= 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += signal[i];
    }
    
    return sum / length;
}