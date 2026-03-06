#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <cmath>

template<typename T>
class Vector {
private:
    T* data;
    size_t size;

public:
    // Constructor
    Vector(size_t n) : size(n) {
        data = new T[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = T();
        }
    }

    // Destructor
    ~Vector() {
        delete[] data;
    }

    // Copy constructor
    Vector(const Vector<T>& other) : size(other.size) {
        data = new T[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }

    // Assignment operator
    Vector<T>& operator=(const Vector<T>& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new T[size];
            for (size_t i = 0; i < size; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Element access operator
    T& operator[](size_t index) {
        return data[index];
    }

    const T& operator[](size_t index) const {
        return data[index];
    }

    // Get size
    size_t getSize() const {
        return size;
    }

    // Dot product
    T dotProduct(const Vector<T>& other) const {
        T result = T();
        for (size_t i = 0; i < size; ++i) {
            result += data[i] * other.data[i];
        }
        return result;
    }

    // Magnitude
    T magnitude() const {
        T sum = T();
        for (size_t i = 0; i < size; ++i) {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }

    // Normalize
    void normalize() {
        T mag = magnitude();
        if (mag != T()) {
            for (size_t i = 0; i < size; ++i) {
                data[i] /= mag;
            }
        }
    }

    // Friend function for vector addition
    template<typename U>
    friend Vector<U> operator+(const Vector<U>& a, const Vector<U>& b);
};

// Vector addition
template<typename T>
Vector<T> operator+(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> result(a.size);
    for (size_t i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

#endif