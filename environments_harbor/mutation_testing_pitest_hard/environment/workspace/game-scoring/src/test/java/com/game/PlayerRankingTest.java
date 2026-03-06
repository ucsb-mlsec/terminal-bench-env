package com.game;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

public class PlayerRankingTest {
    
    private PlayerRanking ranking;
    
    @Before
    public void setUp() {
        ranking = new PlayerRanking();
    }
    
    @Test
    public void testGetTier() {
        String result = ranking.getTier(5000);
        assertEquals("Platinum", result);
    }
    
    @Test
    public void testIsEligibleForPromotion() {
        boolean result = ranking.isEligibleForPromotion(1500, 15);
        assertTrue(result);
    }
    
    @Test
    public void testCalculateRankPoints() {
        int result = ranking.calculateRankPoints(100, 3);
        assertEquals(300, result);
    }
}