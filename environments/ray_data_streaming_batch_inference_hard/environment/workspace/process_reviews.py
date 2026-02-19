#!/usr/bin/env python3

import sys
import time
from sentiment import analyze_sentiment

def main():
    """
    Inefficient implementation that loads all reviews into memory
    and processes them sequentially.
    """
    start_time = time.time()
    
    try:
        # Load ALL reviews into memory at once - inefficient for large datasets
        print("Loading all reviews into memory...")
        with open('/data/reviews.txt', 'r', encoding='utf-8') as f:
            # readlines() loads entire file into memory as a list
            reviews = f.readlines()
        
        total_reviews = len(reviews)
        print(f"Loaded {total_reviews} reviews into memory")
        
        # Process reviews sequentially without parallelization
        print("Processing reviews sequentially...")
        sentiment_scores = []
        
        # Simple for loop - no parallelization, processes one at a time
        for i, review in enumerate(reviews):
            # Strip whitespace and process each review
            review_text = review.strip()
            if review_text:  # Skip empty lines
                # Call analyze_sentiment on each review one at a time
                score = analyze_sentiment(review_text)
                sentiment_scores.append(score)
            
            # Progress indicator every 10000 reviews
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{total_reviews} reviews...")
        
        # Calculate average sentiment score
        if sentiment_scores:
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            average_sentiment = 0.0
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Print results to console
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Total reviews processed: {len(sentiment_scores)}")
        print(f"Average sentiment score: {average_sentiment:.4f}")
        print(f"Processing time: {processing_time:.1f} seconds")
        print("="*50)
        
    except FileNotFoundError:
        print("Error: Could not find /data/reviews.txt", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()