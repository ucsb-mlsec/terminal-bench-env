import os
import pandas as pd
from dagster import asset, Definitions

@asset
def raw_feedback():
    """Stage 1: Read raw customer feedback data"""
    df = pd.read_csv('./data/raw_feedback.csv')
    return df

@asset
def cleaned_feedback(raw_feedback):
    """Stage 2: Clean the feedback data"""
    df = raw_feedback.copy()
    
    # Remove rows with null values
    df = df.dropna()
    
    # Standardize text - convert to lowercase and strip whitespace
    if 'feedback' in df.columns:
        df['feedback'] = df['feedback'].str.lower().str.strip()
    if 'customer_name' in df.columns:
        df['customer_name'] = df['customer_name'].str.strip()
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Save cleaned data
    df.to_csv('./output/cleaned_feedback.csv', index=False)
    
    return df

@asset
def analyzed_feedback(cleaned_feedback):
    """Stage 3: Analyze the feedback data"""
    df = cleaned_feedback.copy()
    
    # Simple sentiment scoring based on keywords
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'happy']
    negative_words = ['bad', 'poor', 'terrible', 'awful', 'worst', 'hate', 'disappointed', 'angry']
    
    def calculate_sentiment(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        positive_count = sum(word in text_lower for word in positive_words)
        negative_count = sum(word in text_lower for word in negative_words)
        return positive_count - negative_count
    
    if 'feedback' in df.columns:
        df['sentiment_score'] = df['feedback'].apply(calculate_sentiment)
        
        # Categorize based on sentiment score
        df['category'] = df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
        )
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Save analyzed data
    df.to_csv('./output/analyzed_feedback.csv', index=False)
    
    return df

defs = Definitions(
    assets=[raw_feedback, cleaned_feedback, analyzed_feedback]
)