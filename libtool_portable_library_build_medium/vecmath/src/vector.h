#ifndef VECTOR_H
#define VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x;
    double y;
    double z;
} Vector3;

extern double vector_dot(Vector3 a, Vector3 b);
extern double vector_magnitude(Vector3 v);
extern Vector3 vector_normalize(Vector3 v);

#ifdef __cplusplus
}
#endif

#endif /* VECTOR_H */