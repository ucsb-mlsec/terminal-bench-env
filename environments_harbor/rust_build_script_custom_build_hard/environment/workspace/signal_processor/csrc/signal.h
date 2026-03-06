#ifndef SIGNAL_H
#define SIGNAL_H

#ifdef __cplusplus
extern "C" {
#endif

void low_pass_filter(double* signal, int length, double cutoff_frequency);

double compute_rms(const double* signal, int length);

double calculate_mean(const double* signal, int length);

void normalize_signal(double* signal, int length);

int detect_peaks(const double* signal, int length, double threshold, int* peak_indices);

#ifdef __cplusplus
}
#endif

#endif