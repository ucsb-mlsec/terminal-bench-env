#include "vector.h"
#include "constants.h"
#include "utils.h"
#include <cstring>

Vector::Vector(int n) : size(n) {
    data = new double[n];
    for (int i = 0; i < n; i++) {
        data[i] = 0.0;
    }
    vectorCounter++;
}

Vector::~Vector() {
    delete[] data;
}

double Vector::get(int i) const {
    return data[i];
}

void Vector::set(int i, double val) {
    data[i] = val;
}

int Vector::getSize() const {
    return size;
}

Vector scaleByPi(const Vector& v) {
    Vector result(v.getSize());
    for (int i = 0; i < v.getSize(); i++) {
        result.set(i, v.get(i) * PI);
    }
    return result;
}

Vector scaleByE(const Vector& v) {
    Vector result(v.getSize());
    for (int i = 0; i < v.getSize(); i++) {
        result.set(i, v.get(i) * EULER_NUMBER);
    }
    return result;
}