#include <math.h>
#include "vector.h"

double vector_dot(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double vector_magnitude(Vector3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vector3 vector_normalize(Vector3 v) {
    double mag = vector_magnitude(v);
    Vector3 result;
    
    if (mag != 0.0) {
        result.x = v.x / mag;
        result.y = v.y / mag;
        result.z = v.z / mag;
    } else {
        result.x = v.x;
        result.y = v.y;
        result.z = v.z;
    }
    
    return result;
}