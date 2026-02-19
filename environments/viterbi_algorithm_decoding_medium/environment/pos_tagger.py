#!/usr/bin/env python3

import json
import sys
import math

def load_model_data(filepath):
    """Load pre-trained HMM model data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['transition_probs'], data['emission_probs'], data['tags']

def tokenize(sentence):
    """Simple tokenization by splitting on whitespace"""
    return sentence.strip().split()

def viterbi_decode(words, tags, transition_probs, emission_probs):
    """
    Viterbi algorithm for finding most probable tag sequence.
    Contains bugs that need to be fixed!
    """
    n_words = len(words)
    n_tags = len(tags)
    
    # BUG 1: Wrong dimensions - should be [n_words][n_tags] but initialized incorrectly
    viterbi = [[0.0 for _ in range(n_words)] for _ in range(n_tags)]
    backpointer = [[0 for _ in range(n_words)] for _ in range(n_tags)]
    
    # Initialization step for first word
    word = words[0]
    for i, tag in enumerate(tags):
        # BUG 2: Missing handling of unknown words
        emission_prob = emission_probs[tag].get(word, 0.0)
        # Initial probability (assume uniform start probability)
        viterbi[i][0] = math.log(emission_prob + 1e-10) + math.log(1.0 / n_tags)
        backpointer[i][0] = 0
    
    # Forward pass through remaining words
    # BUG 3: Off-by-one error in range - should start from 1, not 0
    for t in range(0, n_words):
        word = words[t]
        for i, curr_tag in enumerate(tags):
            emission_prob = emission_probs[curr_tag].get(word, 1e-10)
            max_prob = float('-inf')
            max_prev_tag = 0
            
            for j, prev_tag in enumerate(tags):
                # Get transition probability
                trans_prob = transition_probs[prev_tag].get(curr_tag, 1e-10)
                # BUG 4: Wrong index - should be t-1, not t
                prob = viterbi[j][t] + math.log(trans_prob) + math.log(emission_prob)
                
                if prob > max_prob:
                    max_prob = prob
                    max_prev_tag = j
            
            viterbi[i][t] = max_prob
            backpointer[i][t] = max_prev_tag
    
    # Find best final state
    max_final_prob = float('-inf')
    best_final_tag = 0
    for i in range(n_tags):
        # BUG 5: Wrong index - should be n_words-1
        if viterbi[i][n_words] > max_final_prob:
            max_final_prob = viterbi[i][n_words]
            best_final_tag = i
    
    # Backtrack to find best path
    best_path = [0] * n_words
    best_path[n_words - 1] = best_final_tag
    
    # BUG 6: Wrong range - should go from n_words-2 down to 0
    for t in range(n_words - 1, 0, -1):
        best_path[t - 1] = backpointer[best_path[t]][t]
    
    # Convert indices to tag names
    tag_sequence = [tags[i] for i in best_path]
    return tag_sequence

def main():
    # Load model
    model_path = '/home/user/tagger/model_data.json'
    transition_probs, emission_probs, tags = load_model_data(model_path)
    
    # Read test sentences
    test_file = '/home/user/tagger/test_sentences.txt'
    with open(test_file, 'r') as f:
        sentences = f.readlines()
    
    # Process each sentence
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        words = tokenize(sentence)
        tag_sequence = viterbi_decode(words, tags, transition_probs, emission_probs)
        
        # Format output as word/TAG pairs
        tagged_words = [f"{word}/{tag}" for word, tag in zip(words, tag_sequence)]
        results.append(' '.join(tagged_words))
    
    # Write results
    output_path = '/tmp/results.txt'
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    print(f"Tagging complete. Results written to {output_path}")

if __name__ == '__main__':
    main()