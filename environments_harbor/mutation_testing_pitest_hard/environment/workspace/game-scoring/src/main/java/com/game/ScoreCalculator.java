package com.game;

public class ScoreCalculator {
    
    public int calculateScore(int basePoints, int achievements) {
        if (basePoints < 0 || achievements < 0) {
            return 0;
        }
        return basePoints + achievements;
    }
    
    public int subtractPenalty(int currentScore, int penalty) {
        if (penalty < 0) {
            return currentScore;
        }
        int result = currentScore - penalty;
        return result < 0 ? 0 : result;
    }
    
    public int multiplyScore(int score, int multiplier) {
        if (score < 0 || multiplier < 0) {
            return 0;
        }
        if (multiplier == 0) {
            return 0;
        }
        return score * multiplier;
    }
    
    public int divideScore(int score, int divisor) {
        if (divisor == 0) {
            throw new IllegalArgumentException("Cannot divide by zero");
        }
        if (score < 0 || divisor < 0) {
            return 0;
        }
        return score / divisor;
    }
    
    public int addBonus(int score, int bonus) {
        if (score < 0) {
            return bonus >= 0 ? bonus : 0;
        }
        if (bonus < 0) {
            return score;
        }
        return score + bonus;
    }
}