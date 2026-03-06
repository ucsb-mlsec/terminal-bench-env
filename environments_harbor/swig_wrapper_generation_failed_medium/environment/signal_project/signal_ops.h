#ifndef SIGNAL_OPS_H
#define SIGNAL_OPS_H

double* add_signals(double* sig1, double* sig2, int length);
double* multiply_signal(double* signal, int length, double multiplier);
double mean_signal(double* signal, int length);

#endif