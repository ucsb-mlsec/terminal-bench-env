#ifndef VECTOR_H
#define VECTOR_H

class Vector {
private:
    double* data;
    int size;

public:
    Vector(int n);
    ~Vector();
    double get(int i) const;
    void set(int i, double val);
    int getSize() const;
};

// Global variable definition - causes multiple definition error
int vectorCounter = 0;

// Non-inline function definition in header - causes multiple definition error
double dotProduct(const Vector& a, const Vector& b) {
    vectorCounter++;
    if (a.getSize() != b.getSize()) {
        return 0.0;
    }
    double result = 0.0;
    for (int i = 0; i < a.getSize(); i++) {
        result += a.get(i) * b.get(i);
    }
    return result;
}

#endif // VECTOR_H