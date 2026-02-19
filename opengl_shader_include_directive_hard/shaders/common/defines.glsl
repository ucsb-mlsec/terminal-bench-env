// Common shader definitions and constants
// This file contains shared constants used across multiple shaders

#ifndef COMMON_DEFINES_GLSL
#define COMMON_DEFINES_GLSL

#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define HALF_PI 1.57079632679
#define MAX_LIGHTS 8
#define MAX_BONES 128
#define EPSILON 0.0001

const float GAMMA = 2.2;
const float INV_GAMMA = 0.45454545;
const int MAX_SHADOW_CASCADES = 4;

#endif // COMMON_DEFINES_GLSL