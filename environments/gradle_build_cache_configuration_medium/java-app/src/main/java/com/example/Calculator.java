package com.example;

/**
 * A simple calculator utility class for basic arithmetic operations.
 * This class provides methods for addition, subtraction, multiplication, and division.
 */
public class Calculator {

    /**
     * Adds two integers and returns the result.
     * @param a first integer
     * @param b second integer
     * @return sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }

    /**
     * Subtracts the second integer from the first and returns the result.
     * @param a first integer
     * @param b second integer
     * @return difference of a and b
     */
    public int subtract(int a, int b) {
        return a - b;
    }

    /**
     * Multiplies two integers and returns the result.
     * @param a first integer
     * @param b second integer
     * @return product of a and b
     */
    public int multiply(int a, int b) {
        return a * b;
    }

    /**
     * Divides the first integer by the second and returns the result.
     * @param a dividend
     * @param b divisor
     * @return quotient of a divided by b
     * @throws ArithmeticException if b is zero
     */
    public double divide(int a, int b) {
        if (b == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return (double) a / b;
    }
}