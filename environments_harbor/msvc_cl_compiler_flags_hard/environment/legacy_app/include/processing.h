#ifndef PROCESSING_H
#define PROCESSING_H

#include <stdint.h>
#include <stddef.h>

/* Data structure for processing operations */
typedef struct {
    int32_t value;
    uint32_t flags;
    double coefficient;
} ProcessData;

/* Data structure for filter parameters */
typedef struct {
    double threshold;
    int samples;
    uint32_t mode;
} FilterParams;

/* Main processing function - processes array of data */
int process_data(ProcessData* data, size_t count);

/* Transform input values according to specified mode */
int transform_input(int32_t* input, size_t length, uint32_t mode);

/* Compute result from processed data */
double compute_result(const ProcessData* data, size_t count);

/* Apply filter with given parameters */
int apply_filter(ProcessData* data, size_t count, const FilterParams* params);

/* Helper function for validation */
int validate_data(const ProcessData* data, size_t count);

#endif /* PROCESSING_H */