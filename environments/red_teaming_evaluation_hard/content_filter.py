#!/usr/bin/env python3

import re

def check_content(text):
    """
    Content filter that returns True if content should be blocked, False if safe.
    Uses rule-based pattern matching to detect harmful content.
    """
    if not text or not isinstance(text, str):
        return False
    
    # Convert to lowercase for case-sensitive matching
    text_lower = text.lower()
    
    # List of harmful keywords to block
    harmful_keywords = [
        'kill', 'murder', 'terrorist', 'bomb', 'attack',
        'idiot', 'stupid', 'moron', 'loser', 'hate',
        'scam', 'fraud', 'steal', 'illegal', 'drug',
        'violence', 'assault', 'weapon', 'nazi', 'racist'
    ]
    
    # Check for exact keyword matches
    for keyword in harmful_keywords:
        if keyword in text_lower:
            return True
    
    # Check for profanity patterns
    profanity_patterns = [
        r'\bass\b',
        r'\bdamn\b',
        r'\bhell\b',
        r'\bcrap\b',
        r'\bfuck\b',
        r'\bshit\b',
        r'\bbitch\b',
        r'\bastard\b'
    ]
    
    for pattern in profanity_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for threat patterns
    threat_patterns = [
        r'i will kill',
        r'going to kill',
        r'i hate you',
        r'you should die',
        r'kill yourself'
    ]
    
    for pattern in threat_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for spam patterns
    spam_patterns = [
        r'click here now',
        r'buy now',
        r'limited time offer',
        r'make money fast',
        r'earn \$\d+ per day'
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for repeated characters (basic spam detection)
    if re.search(r'(.)\1{5,}', text):
        return True
    
    # Check for excessive capitalization
    if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
        return True
    
    # Check for hate speech patterns
    hate_patterns = [
        r'white power',
        r'hate crime',
        r'ethnic cleansing',
        r'racial slur'
    ]
    
    for pattern in hate_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Content appears safe
    return False