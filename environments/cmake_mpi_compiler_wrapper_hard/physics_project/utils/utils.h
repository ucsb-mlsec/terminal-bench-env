#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the magnitude of a 3D vector */
double vector_magnitude(double x, double y, double z);

/* Initialize an array with default values */
void initialize_array(double* arr, int size);

/* Print a standard header for the physics simulation */
void print_header(void);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */