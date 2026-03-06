#ifndef COUNTER_H
#define COUNTER_H

#include <cstdint>

struct Counter {
    uint64_t count;
    uint64_t errors;
    double sum;
};

#endif // COUNTER_H