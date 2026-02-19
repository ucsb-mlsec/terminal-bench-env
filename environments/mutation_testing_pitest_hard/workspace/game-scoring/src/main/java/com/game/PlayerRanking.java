package com.game;

public class PlayerRanking {
    
    public String getTier(int score) {
        if (score < 0) {
            throw new IllegalArgumentException("Score cannot be negative");
        }
        
        if (score < 1000) {
            return "Bronze";
        } else if (score < 2500) {
            return "Silver";
        } else if (score < 5000) {
            return "Gold";
        } else if (score < 10000) {
            return "Platinum";
        } else {
            return "Diamond";
        }
    }
    
    public boolean isEligibleForPromotion(int currentScore, int wins) {
        if (currentScore < 0) {
            throw new IllegalArgumentException("Score cannot be negative");
        }
        if (wins < 0) {
            throw new IllegalArgumentException("Wins cannot be negative");
        }
        
        return currentScore >= 1000 && wins >= 10;
    }
    
    public int calculateRankPoints(int score, int level) {
        if (score < 0) {
            throw new IllegalArgumentException("Score cannot be negative");
        }
        if (level < 0) {
            throw new IllegalArgumentException("Level cannot be negative");
        }
        
        if (level < 5) {
            return score * 1;
        } else if (level < 10) {
            return score * 2;
        } else {
            return score * 3;
        }
    }
    
    public boolean needsRankProtection(int consecutiveLosses, int currentTier) {
        if (consecutiveLosses < 0) {
            throw new IllegalArgumentException("Consecutive losses cannot be negative");
        }
        if (currentTier < 1 || currentTier > 5) {
            throw new IllegalArgumentException("Current tier must be between 1 and 5");
        }
        
        return consecutiveLosses >= 3 && currentTier >= 3;
    }
}