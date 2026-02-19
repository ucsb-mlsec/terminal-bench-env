#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <memory>

#define PI 3.14159
#define VERSION "1.0.0"

const int MAX_SIZE = 100;
const double EPSILON = 1e-6;

inline bool isZero(double value) {
    return std::abs(value) < EPSILON;
}

#endif