package com.game;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

public class ScoreCalculatorTest {
    
    private ScoreCalculator calculator;
    
    @Before
    public void setUp() {
        calculator = new ScoreCalculator();
    }
    
    @Test
    public void testCalculateScore() {
        int result = calculator.calculateScore(100, 50);
        assertEquals(150, result);
    }
    
    @Test
    public void testSubtractPenalty() {
        int result = calculator.subtractPenalty(100, 30);
        assertEquals(70, result);
    }
    
    @Test
    public void testMultiplyScore() {
        int result = calculator.multiplyScore(10, 5);
        assertEquals(50, result);
    }
}