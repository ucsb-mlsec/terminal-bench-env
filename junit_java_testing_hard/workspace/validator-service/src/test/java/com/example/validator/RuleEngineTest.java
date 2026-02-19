package com.example.validator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.Assert;

public class RuleEngineTest {
    
    private RuleEngine ruleEngine;
    
    @BeforeEach
    public void setUp() {
        ruleEngine = new RuleEngine();
    }
    
    @Test
    public void testCalculateScore() {
        int result = ruleEngine.calculateScore(10);
        Assert.assertEquals(20, result);
    }
    
    @Test
    public void testFormatResult() {
        String result = ruleEngine.formatResult("hello");
        Assert.assertEquals("HELLO", result);
    }
    
    @Test
    public void testEvaluateCondition() {
        boolean result = ruleEngine.evaluateCondition(true, true);
        Assert.assertTrue(result);
    }
}