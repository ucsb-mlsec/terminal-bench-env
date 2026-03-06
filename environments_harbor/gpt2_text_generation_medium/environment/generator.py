#!/usr/bin/env python3

import os
import random
import hashlib

def read_prompts(filepath):
    """Read prompts from file, one per line."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def read_patterns(pattern_dir):
    """Read all pattern files from the patterns directory."""
    patterns = []
    if os.path.exists(pattern_dir):
        for filename in os.listdir(pattern_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(pattern_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        patterns.append(content)
    return patterns

def generate_text(prompt, patterns, min_words=50):
    """Generate text continuation based on prompt and patterns."""
    if not patterns:
        # Fallback if no patterns available
        patterns = [
            "The journey continues with unexpected discoveries and challenges ahead.",
            "This leads to new opportunities and experiences that shape the future.",
            "Through careful consideration and planning, progress is made steadily.",
            "The outcome reveals important insights and valuable lessons learned.",
            "With determination and effort, goals are achieved step by step."
        ]
    
    # Use prompt as seed for deterministic generation
    seed = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    
    # Generate text by combining patterns
    generated = []
    word_count = 0
    
    # Shuffle patterns deterministically based on prompt
    available_patterns = patterns.copy()
    rng.shuffle(available_patterns)
    
    # Keep adding patterns until we reach minimum word count
    pattern_index = 0
    while word_count < min_words:
        pattern = available_patterns[pattern_index % len(available_patterns)]
        generated.append(pattern)
        word_count += len(pattern.split())
        pattern_index += 1
        
        # Add connecting words between patterns
        if word_count < min_words and pattern_index < 10:
            connectors = [
                "Furthermore,", "Additionally,", "Moreover,", "In addition,",
                "As a result,", "Consequently,", "Therefore,", "Thus,"
            ]
            connector = rng.choice(connectors)
            generated.append(connector)
            word_count += 1
    
    return ' '.join(generated)

def main():
    prompts_file = '/workspace/prompts.txt'
    patterns_dir = '/workspace/patterns/'
    output_file = '/workspace/results.txt'
    
    # Read prompts and patterns
    prompts = read_prompts(prompts_file)
    patterns = read_patterns(patterns_dir)
    
    # Generate text for each prompt
    results = []
    for i, prompt in enumerate(prompts, 1):
        results.append(f"=== PROMPT {i} ===")
        results.append(prompt)
        results.append("")
        generated = generate_text(prompt, patterns)
        results.append(generated)
        results.append("")
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))
    
    print(f"Generated text for {len(prompts)} prompts")
    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()