package com.game;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

public class BonusMultiplierTest {
    
    private BonusMultiplier multiplier;
    
    @Before
    public void setUp() {
        multiplier = new BonusMultiplier();
    }
    
    @Test
    public void testApplyMultiplier() {
        int result = multiplier.applyMultiplier(100, true, true);
        assertEquals(300, result);
    }
    
    @Test
    public void testGetStreakBonus() {
        int result = multiplier.getStreakBonus(5);
        assertEquals(50, result);
    }
    
    @Test
    public void testCalculateComboMultiplier() {
        double result = multiplier.calculateComboMultiplier(4);
        assertEquals(2.0, result, 0.01);
    }
}