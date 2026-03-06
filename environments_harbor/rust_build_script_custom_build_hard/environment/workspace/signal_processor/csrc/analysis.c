#include "signal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double calculate_rms(const double* signal, int length) {
    if (signal == NULL || length <= 0) {
        return 0.0;
    }
    
    double sum_of_squares = 0.0;
    
    for (int i = 0; i < length; i++) {
        sum_of_squares += signal[i] * signal[i];
    }
    
    double mean_square = sum_of_squares / length;
    double rms = sqrt(mean_square);
    
    return rms;
}

double calculate_mean(const double* signal, int length) {
    if (signal == NULL || length <= 0) {
        return 0.0;
    }
    
    double sum = 0.0;
    
    for (int i = 0; i < length; i++) {
        sum += signal[i];
    }
    
    return sum / length;
}