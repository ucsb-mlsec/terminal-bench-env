#!/usr/bin/env python3
"""
Legacy text analyzer with hard-coded 512-character limit.
DO NOT MODIFY - This is a critical legacy component.
"""

def analyze_text(text):
    """
    Analyze text and return category scores.
    
    WARNING: This is a legacy system with a 512-character limit.
    Cannot be modified due to system constraints.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Category scores (0.0 to 1.0) for 'technical', 'business', 'support'
        
    Raises:
        ValueError: If text exceeds 512 characters
    """
    # Hard-coded limit check
    if len(text) > 512:
        raise ValueError("Text exceeds maximum length of 512 characters")
    
    # Handle empty text
    if not text or len(text) == 0:
        return {'technical': 0.0, 'business': 0.0, 'support': 0.0}
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Define keywords for each category
    technical_keywords = ['system', 'software', 'code', 'algorithm', 'data', 
                         'network', 'security', 'api']
    business_keywords = ['revenue', 'profit', 'sales', 'market', 'customer', 
                        'strategy', 'growth', 'investment']
    support_keywords = ['help', 'issue', 'problem', 'assist', 'question', 
                       'ticket', 'resolve', 'troubleshoot']
    
    # Count keyword occurrences
    technical_count = sum(text_lower.count(keyword) for keyword in technical_keywords)
    business_count = sum(text_lower.count(keyword) for keyword in business_keywords)
    support_count = sum(text_lower.count(keyword) for keyword in support_keywords)
    
    # Normalize by text length (use character count for normalization)
    # Scale factor to make scores reasonable
    scale_factor = 100.0
    text_length = len(text)
    
    technical_score = min(1.0, (technical_count * scale_factor) / text_length)
    business_score = min(1.0, (business_count * scale_factor) / text_length)
    support_score = min(1.0, (support_count * scale_factor) / text_length)
    
    return {
        'technical': round(technical_score, 2),
        'business': round(business_score, 2),
        'support': round(support_score, 2)
    }
```

```python
#!/usr/bin/env python3
"""
Document processing script that handles documents of any length
by working around the 512-character limitation of the legacy analyzer.
"""

import sys
import json
import os
import argparse
from pathlib import Path

# Add analyzer directory to path
sys.path.insert(0, '/workspace/analyzer')
from text_analyzer import analyze_text


def chunk_text(text, chunk_size=512):
    """
    Split text into chunks of specified size.
    
    Args:
        text (str): Input text
        chunk_size (int): Maximum size of each chunk
        
    Returns:
        list: List of text chunks
    """
    if not text:
        return ['']
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    
    return chunks


def analyze_document(content):
    """
    Analyze a document of any length by chunking it and aggregating results.
    
    Args:
        content (str): Document content
        
    Returns:
        dict: Aggregated category scores
    """
    # Handle empty content
    if not content:
        return analyze_text('')
    
    # Split content into chunks
    chunks = chunk_text(content, chunk_size=512)
    
    # Analyze each chunk
    all_scores = []
    for chunk in chunks:
        try:
            scores = analyze_text(chunk)
            all_scores.append(scores)
        except ValueError as e:
            # Should not happen with proper chunking, but handle gracefully
            continue
    
    # If no chunks were successfully analyzed, return zero scores
    if not all_scores:
        return {'technical': 0.0, 'business': 0.0, 'support': 0.0}
    
    # Aggregate scores by averaging
    categories = ['technical', 'business', 'support']
    aggregated_scores = {}
    
    for category in categories:
        category_scores = [scores[category] for scores in all_scores]
        aggregated_scores[category] = sum(category_scores) / len(category_scores)
    
    return aggregated_scores


def get_top_category(scores, valid_categories):
    """
    Get the category with the highest score.
    
    Args:
        scores (dict): Category scores
        valid_categories (set): Set of valid category names
        
    Returns:
        str: Category name with highest score
    """
    # Filter to only valid categories
    valid_scores = {k: v for k, v in scores.items() if k in valid_categories}
    
    if not valid_scores:
        # Fallback to first valid category if no valid scores
        return sorted(valid_categories)[0]
    
    # Find maximum score
    max_score = max(valid_scores.values())
    
    # Get all categories with max score
    top_categories = [cat for cat, score in valid_scores.items() if score == max_score]
    
    # If tie, sort alphabetically and take first
    return sorted(top_categories)[0]


def load_valid_categories(categories_file):
    """
    Load valid category names from file.
    
    Args:
        categories_file (str): Path to categories file
        
    Returns:
        set: Set of valid category names
    """
    categories = set()
    
    if os.path.exists(categories_file):
        with open(categories_file, 'r') as f:
            for line in f:
                category = line.strip()
                if category:
                    categories.add(category)
    
    # Fallback to default categories if file not found or empty
    if not categories:
        categories = {'technical', 'business', 'support'}
    
    return categories


def process_documents(input_file, output_file, categories_file):
    """
    Process all documents and generate results.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output results file
        categories_file (str): Path to categories file
    """
    # Load valid categories
    valid_categories = load_valid_categories(categories_file)
    
    # Process documents
    results = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                doc = json.loads(line)
                doc_id = doc['doc_id']
                content = doc['content']
                
                # Analyze document
                scores = analyze_document(content)
                
                # Get top category
                category = get_top_category(scores, valid_categories)
                
                # Store result
                results.append((doc_id, category))
                
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed lines
                continue
    
    # Write results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for doc_id, category in results:
            f.write(f"{doc_id}={category}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process documents with legacy analyzer')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = '/workspace/solution/results.txt'
    categories_file = '/workspace/data/categories.txt'
    
    process_documents(input_file, output_file, categories_file)


if __name__ == '__main__':
    main()