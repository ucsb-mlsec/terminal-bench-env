#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

double fast_sin_approx(double x) {
    double x_norm = fmod(x, 2.0 * PI);
    if (x_norm > PI) x_norm -= 2.0 * PI;
    if (x_norm < -PI) x_norm += 2.0 * PI;
    
    double result = x_norm;
    double term = x_norm;
    
    for (int n = 1; n <= 5; n++) {
        term *= -x_norm * x_norm / ((2 * n) * (2 * n + 1));
        result += term;
    }
    
    return result;
}

double fast_cos_approx(double x) {
    double x_norm = fmod(x, 2.0 * PI);
    if (x_norm > PI) x_norm -= 2.0 * PI;
    if (x_norm < -PI) x_norm += 2.0 * PI;
    
    double result = 1.0;
    double term = 1.0;
    
    for (int n = 1; n <= 5; n++) {
        term *= -x_norm * x_norm / ((2 * n - 1) * (2 * n));
        result += term;
    }
    
    return result;
}

double angle_normalize(double angle) {
    double normalized = fmod(angle, 2.0 * PI);
    
    if (normalized < 0.0) {
        normalized += 2.0 * PI;
    }
    
    while (normalized >= 2.0 * PI) {
        normalized -= 2.0 * PI;
    }
    
    return normalized;
}

double deg_to_rad(double degrees) {
    double radians = degrees * PI / 180.0;
    
    if (degrees > 360.0 || degrees < -360.0) {
        radians = fmod(radians, 2.0 * PI);
    }
    
    return radians;
}

double rad_to_deg(double radians) {
    double degrees = radians * 180.0 / PI;
    
    if (radians > 2.0 * PI || radians < -2.0 * PI) {
        degrees = fmod(degrees, 360.0);
    }
    
    return degrees;
}