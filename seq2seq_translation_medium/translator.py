#!/usr/bin/env python3

import json
import sys
import os
import re

def load_dictionary(filepath='/workspace/dictionary.json'):
    """Load the English-Spanish dictionary from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        return dictionary
    except FileNotFoundError:
        print(f"Error: Dictionary file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in dictionary file")
        sys.exit(1)

def load_grammar_rules(filepath='/workspace/grammar_rules.json'):
    """Load grammar rules from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        return rules
    except FileNotFoundError:
        print(f"Error: Grammar rules file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in grammar rules file")
        sys.exit(1)

def tokenize_sentence(sentence):
    """Split sentence into words and handle punctuation"""
    # Remove extra whitespace
    sentence = sentence.strip()
    
    # Split by spaces but preserve punctuation
    tokens = []
    current_word = ""
    
    for char in sentence:
        if char.isalpha() or char == "'":
            current_word += char
        elif char in [' ', '\t', '\n']:
            if current_word:
                tokens.append(current_word)
                current_word = ""
        else:
            if current_word:
                tokens.append(current_word)
                current_word = ""
            if char.strip():  # Non-whitespace punctuation
                tokens.append(char)
    
    if current_word:
        tokens.append(current_word)
    
    return tokens

def translate_word(word, dictionary):
    """Translate a single word using the dictionary"""
    # BUG 1: Not converting to lowercase for dictionary lookup
    if word in dictionary:
        return dictionary[word]
    else:
        # Return the original word if not found
        return word

def get_noun_gender(noun, grammar_rules):
    """Determine the gender of a noun"""
    if 'noun_genders' in grammar_rules:
        # BUG 2: Using wrong key 'gender' instead of actual noun
        if noun in grammar_rules['noun_genders']:
            return grammar_rules['noun_genders'][noun]
    return 'masculine'  # Default to masculine

def apply_article(article, noun, grammar_rules):
    """Apply correct article form based on noun gender"""
    gender = get_noun_gender(noun, grammar_rules)
    
    article_lower = article.lower()
    
    if article_lower == 'the':
        if gender == 'feminine':
            return 'la'
        else:
            return 'el'
    elif article_lower == 'a' or article_lower == 'an':
        if gender == 'feminine':
            return 'una'
        else:
            return 'un'
    
    return article

def apply_grammar_rules(tokens, translated_tokens, grammar_rules):
    """Apply Spanish grammar rules to translated tokens"""
    result = []
    i = 0
    
    while i < len(translated_tokens):
        current = translated_tokens[i]
        original = tokens[i]
        
        # Check if current token is an article
        if original.lower() in ['the', 'a', 'an']:
            # BUG 3: Off-by-one error - checking i+1 but not verifying bounds properly
            if i + 1 <= len(translated_tokens):
                next_token = translated_tokens[i + 1]
                original_next = tokens[i + 1]
                
                # Apply correct article
                correct_article = apply_article(original, next_token, grammar_rules)
                result.append(correct_article)
                i += 1
                continue
        
        # Check for adjective-noun reordering
        if i > 0 and i < len(tokens):
            prev_original = tokens[i - 1].lower()
            current_original = original.lower()
            
            # If previous was an adjective and current is a noun, they should be swapped
            if 'adjectives' in grammar_rules:
                if prev_original in grammar_rules['adjectives']:
                    # BUG 4: Wrong order - should put noun first, then adjective
                    # This bug causes adjectives to stay before nouns (English order)
                    pass
        
        result.append(current)
        i += 1
    
    return result

def translate_sentence(sentence, dictionary, grammar_rules):
    """Translate a complete English sentence to Spanish"""
    if not sentence or sentence.strip() == "":
        return ""
    
    # Tokenize the sentence
    tokens = tokenize_sentence(sentence)
    
    if not tokens:
        return ""
    
    # Translate each word
    translated_tokens = []
    for token in tokens:
        translated = translate_word(token, dictionary)
        translated_tokens.append(translated)
    
    # Apply grammar rules
    final_tokens = apply_grammar_rules(tokens, translated_tokens, grammar_rules)
    
    # Reconstruct sentence
    result = ""
    for i, token in enumerate(final_tokens):
        if i == 0:
            result = token.capitalize()
        elif token in ['.', ',', '!', '?', ';', ':']:
            result += token
        else:
            result += " " + token
    
    return result

def process_test_file(input_file, output_file, dictionary, grammar_rules):
    """Process all sentences in the test input file"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        
        translations = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                translation = translate_sentence(sentence, dictionary, grammar_rules)
                translations.append(translation)
        
        # BUG 5: Wrong file path variable used
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        
        print(f"Translation complete. Results written to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    """Main execution function"""
    # Load dictionary and grammar rules
    dictionary = load_dictionary()
    grammar_rules = load_grammar_rules()
    
    # Define input and output files
    input_file = '/workspace/test_input.txt'
    output_file = '/workspace/solution.txt'
    
    # Process the test file
    process_test_file(input_file, output_file, dictionary, grammar_rules)

if __name__ == "__main__":
    main()