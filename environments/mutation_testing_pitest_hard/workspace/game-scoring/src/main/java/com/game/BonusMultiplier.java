package com.game;

public class BonusMultiplier {
    
    public int applyMultiplier(int baseScore, boolean isWeekend, boolean isPremium) {
        if (isWeekend && isPremium) {
            return baseScore * 3;
        } else if (isWeekend) {
            return baseScore * 2;
        } else if (isPremium) {
            return (int) (baseScore * 1.5);
        } else {
            return baseScore;
        }
    }
    
    public int getStreakBonus(int consecutiveWins) {
        if (consecutiveWins < 3) {
            return 0;
        } else if (consecutiveWins >= 3 && consecutiveWins <= 5) {
            return 50;
        } else if (consecutiveWins >= 6 && consecutiveWins <= 9) {
            return 100;
        } else {
            return 200;
        }
    }
    
    public double calculateComboMultiplier(int comboCount) {
        if (comboCount >= 0 && comboCount <= 1) {
            return 1.0;
        } else if (comboCount >= 2 && comboCount <= 3) {
            return 1.5;
        } else if (comboCount >= 4 && comboCount <= 5) {
            return 2.0;
        } else {
            return 2.5;
        }
    }
    
    public int applyTimeBonus(int score, int hoursPlayed) {
        if (hoursPlayed < 1) {
            return score;
        } else if (hoursPlayed >= 1 && hoursPlayed <= 3) {
            return score + (score * 10 / 100);
        } else if (hoursPlayed >= 4 && hoursPlayed <= 6) {
            return score + (score * 25 / 100);
        } else {
            return score + (score * 50 / 100);
        }
    }
}