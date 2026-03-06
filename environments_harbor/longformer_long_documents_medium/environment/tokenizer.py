#!/usr/bin/env python3

import os
import json
import math
from tokenizer import count_tokens

def analyze_documents(documents_dir, max_tokens=4096):
    """
    Analyze all documents in the given directory and determine chunking requirements.
    
    Args:
        documents_dir: Path to directory containing text files
        max_tokens: Maximum tokens allowed per chunk
        
    Returns:
        List of dictionaries containing analysis results
    """
    results = []
    
    # Get all .txt files and sort them alphabetically
    files = sorted([f for f in os.listdir(documents_dir) if f.endswith('.txt')])
    
    for filename in files:
        filepath = os.path.join(documents_dir, filename)
        
        # Read the document
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count tokens
        total_tokens = count_tokens(content)
        
        # Calculate chunks needed
        # Use ceiling division to ensure no chunk exceeds the limit
        chunks_needed = math.ceil(total_tokens / max_tokens)
        
        # Ensure at least 1 chunk
        chunks_needed = max(1, chunks_needed)
        
        results.append({
            "filename": filename,
            "total_tokens": total_tokens,
            "chunks_needed": chunks_needed
        })
    
    return results

def main():
    documents_dir = '/workspace/documents/'
    output_file = '/workspace/chunk_plan.json'
    
    # Analyze all documents
    analysis_results = analyze_documents(documents_dir)
    
    # Write results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_file}")
    print(f"Processed {len(analysis_results)} documents")

if __name__ == "__main__":
    main()