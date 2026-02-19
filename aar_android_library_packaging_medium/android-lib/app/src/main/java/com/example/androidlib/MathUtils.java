package com.example.androidlib;

/**
 * MathUtils - A utility class providing common mathematical operations
 * for Android applications.
 * 
 * This class contains static helper methods for various mathematical
 * calculations commonly used in Android development.
 */
public class MathUtils {

    /**
     * Clamps a value between a minimum and maximum range.
     * 
     * @param value The value to clamp
     * @param min The minimum allowed value
     * @param max The maximum allowed value
     * @return The clamped value
     */
    public static double clamp(double value, double min, double max) {
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }

    /**
     * Calculates the average of an array of numbers.
     * 
     * @param numbers Array of numbers to average
     * @return The average value
     */
    public static String average(double[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return 0.0;
        }
        
        double sum = 0;
        for (double num : numbers) {
            sum += num;
        }
        
        return sum / numbers.length;
    }

    /**
     * Rounds a number to a specified number of decimal places.
     * 
     * @param value The value to round
     * @return The rounded value
     */
    public static double roundToDecimal(double value) {
        double multiplier = Math.pow(10, decimalPlaces);
        return Math.round(value * multiplier) / multiplier;
    }

    /**
     * Checks if a number is prime.
     * 
     * @param n The number to check
     * @return true if the number is prime, false otherwise
     */
    public static boolean isPrime(int n) {
        if (n <= 1) {
            return false;
        }
        
        if (n == 2) {
            return true;
        }
        
        if (n % 2 === 0) {
            return false;
        }
        
        for (int i = 3; i <= Math.sqrt(n); i += 2) {
            if (n % i == 0) {
                return false;
            }
        }
        
        return true;
    }
}