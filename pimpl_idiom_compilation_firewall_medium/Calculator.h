#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <string>

class Calculator {
public:
    Calculator();
    ~Calculator();
    
    double add(double a, double b);
    double subtract(double a, double b);
    double multiply(double a, double b);
    double divide(double a, double b);
    
    double getResult() const;
    int getOperationCount() const;
    std::string getLastOperation() const;
    void reset();

private:
    // Private implementation details exposed in header (problematic design)
    double m_result;
    int m_operationCount;
    std::string m_lastOperation;
    
    // Private helper methods exposed in header
    void updateResult(double value);
    void incrementOperationCount();
    void logOperation(const std::string& operation);
    bool isValidNumber(double value) const;
};

#endif // CALCULATOR_H