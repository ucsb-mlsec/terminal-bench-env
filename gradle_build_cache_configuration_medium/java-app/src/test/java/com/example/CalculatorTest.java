package com.example;

import org.junit.Test;
import org.junit.Before;
import org.junit.Assert;

public class CalculatorTest {
    
    private Calculator calculator;
    
    @Before
    public void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    public void testAddition() {
        int result = calculator.add(5, 3);
        Assert.assertEquals(8, result);
    }
    
    @Test
    public void testSubtraction() {
        int result = calculator.subtract(10, 4);
        Assert.assertEquals(6, result);
    }
    
    @Test
    public void testMultiplication() {
        int result = calculator.multiply(6, 7);
        Assert.assertEquals(42, result);
    }
    
    @Test
    public void testDivision() {
        int result = calculator.divide(20, 4);
        Assert.assertEquals(5, result);
    }
    
    @Test
    public void testAdditionWithNegativeNumbers() {
        int result = calculator.add(-5, 3);
        Assert.assertEquals(-2, result);
    }
}