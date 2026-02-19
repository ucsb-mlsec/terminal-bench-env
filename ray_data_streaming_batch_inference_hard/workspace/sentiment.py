#!/usr/bin/env python3

import time
import re

POSITIVE_WORDS = {
    'excellent', 'great', 'amazing', 'love', 'best', 'wonderful', 
    'fantastic', 'perfect', 'outstanding', 'awesome', 'good', 
    'beautiful', 'brilliant', 'superb', 'delighted', 'happy',
    'satisfied', 'recommend', 'impressed', 'quality'
}

NEGATIVE_WORDS = {
    'terrible', 'awful', 'hate', 'worst', 'horrible', 'bad', 
    'disappointing', 'poor', 'useless', 'waste', 'broken',
    'defective', 'frustrated', 'angry', 'refund', 'return',
    'never', 'avoid', 'unhappy', 'dissatisfied'
}

def analyze_sentiment(review_text):
    """
    Analyze sentiment of review text and return a score between 0.0 and 1.0.
    
    This function performs sentiment analysis by:
    1. Converting text to lowercase
    2. Counting occurrences of positive and negative words
    3. Calculating a base score from word sentiment balance
    4. Adjusting slightly based on review length
    5. Clamping result between 0.0 and 1.0
    
    Args:
        review_text (str): The review text to analyze
        
    Returns:
        float: Sentiment score between 0.0 (very negative) and 1.0 (very positive),
               rounded to 4 decimal places
    """
    if not review_text or not isinstance(review_text, str):
        return 0.5
    
    # Convert to lowercase for case-insensitive matching
    text_lower = review_text.lower()
    
    # Extract words using regex (simulates realistic processing time)
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count positive and negative word occurrences
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    # Calculate base score
    sentiment_diff = positive_count - negative_count
    base_score = 0.5 + (sentiment_diff * 0.1)
    
    # Add slight variance based on review length
    # Longer reviews with no sentiment words trend slightly positive
    word_count = len(words)
    if word_count > 0 and positive_count == 0 and negative_count == 0:
        length_adjustment = min(0.05, word_count / 1000)
        base_score += length_adjustment
    
    # Additional processing to simulate realistic computation time
    # Count characters and perform boundary checks
    char_count = len(review_text)
    has_exclamation = '!' in review_text
    has_question = '?' in review_text
    
    # Minor adjustments based on punctuation
    if has_exclamation and positive_count > 0:
        base_score += 0.02
    elif has_exclamation and negative_count > 0:
        base_score -= 0.02
    
    # Clamp between 0.0 and 1.0
    final_score = max(0.0, min(1.0, base_score))
    
    # Round to 4 decimal places
    return round(final_score, 4)