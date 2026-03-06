#ifndef STATISTICS_H
#define STATISTICS_H

#include "vector.h"

// Global variable definition - causes multiple definition error
int statsCallCount = 0;

class Stats {
public:
    static double mean(const Vector& v);
    static double variance(const Vector& v);
    static double stddev(const Vector& v);
};

// Non-inline helper function definition in header - causes multiple definition error
double sumElements(const Vector& v) {
    double sum = 0.0;
    for (int i = 0; i < v.size(); ++i) {
        sum += v.get(i);
    }
    return sum;
}

#endif // STATISTICS_H