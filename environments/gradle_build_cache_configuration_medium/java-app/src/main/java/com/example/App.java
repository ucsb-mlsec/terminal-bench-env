/**
 * Main application class for demonstrating Gradle build caching.
 * This class provides basic functionality including string manipulation
 * and arithmetic operations.
 * 
 * @author Development Team
 * @version 1.0
 */
package com.example;

public class App {
    
    /**
     * Main entry point for the application.
     * Demonstrates basic functionality and prints results.
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        
        // Perform some basic computations
        int result = calculateSum(10, 20, 30);
        System.out.println("Sum result: " + result);
        
        String formatted = formatMessage("Gradle", "Build Cache");
        System.out.println(formatted);
        
        double average = calculateAverage(15, 25, 35, 45);
        System.out.println("Average: " + average);
    }
    
    /**
     * Calculates the sum of multiple integers.
     * 
     * @param numbers Variable number of integers to sum
     * @return The sum of all provided numbers
     */
    public static int calculateSum(int... numbers) {
        int sum = 0;
        for (int num : numbers) {
            sum += num;
        }
        return sum;
    }
    
    /**
     * Formats a message by combining two strings with a separator.
     * 
     * @param first The first part of the message
     * @param second The second part of the message
     * @return A formatted string combining both inputs
     */
    public static String formatMessage(String first, String second) {
        return String.format("Welcome to %s with %s enabled!", first, second);
    }
    
    /**
     * Calculates the average of multiple numbers.
     * 
     * @param numbers Variable number of integers
     * @return The average value as a double
     */
    public static double calculateAverage(int... numbers) {
        if (numbers.length == 0) {
            return 0.0;
        }
        return (double) calculateSum(numbers) / numbers.length;
    }
}