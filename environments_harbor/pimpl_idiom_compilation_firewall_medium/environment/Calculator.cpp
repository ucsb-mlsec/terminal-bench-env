#include "Calculator.h"
#include <stdexcept>

Calculator::Calculator() : m_result(0.0), m_operationCount(0), m_lastOperation("none") {
}

Calculator::~Calculator() {
}

double Calculator::add(double a, double b) {
    m_result = a + b;
    m_operationCount++;
    m_lastOperation = "addition";
    return m_result;
}

double Calculator::subtract(double a, double b) {
    m_result = a - b;
    m_operationCount++;
    m_lastOperation = "subtraction";
    return m_result;
}

double Calculator::multiply(double a, double b) {
    m_result = a * b;
    m_operationCount++;
    m_lastOperation = "multiplication";
    return m_result;
}

double Calculator::divide(double a, double b) {
    if (b == 0.0) {
        m_result = 0.0;
        m_operationCount++;
        m_lastOperation = "division";
        return 0.0;
    }
    m_result = a / b;
    m_operationCount++;
    m_lastOperation = "division";
    return m_result;
}

double Calculator::getResult() const {
    return m_result;
}

int Calculator::getOperationCount() const {
    return m_operationCount;
}

void Calculator::reset() {
    m_result = 0.0;
    m_operationCount = 0;
    m_lastOperation = "none";
}