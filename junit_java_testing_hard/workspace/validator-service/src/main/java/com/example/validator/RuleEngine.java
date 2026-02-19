package com.example.validator;

public class RuleEngine {
    
    public RuleEngine() {
        // Default constructor
    }
    
    public int calculateScore(int baseValue) {
        return baseValue * 2;
    }
    
    public String formatResult(String input) {
        return input.toUpperCase();
    }
    
    public boolean evaluateCondition(boolean condition1, boolean condition2) {
        return condition1 && condition2;
    }
}