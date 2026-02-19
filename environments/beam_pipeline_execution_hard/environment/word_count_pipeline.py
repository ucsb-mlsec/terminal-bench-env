#!/usr/bin/env python3

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, SetupOptions
import json
import re
import sys
import os
from typing import Tuple, Dict, List, Any
from collections import defaultdict
import pickle
import time

# This pipeline processes text files for word counting
# TODO: Performance optimization needed - running slow on large files

class WordExtractionDoFn(beam.DoFn):
    """
    Custom DoFn for extracting words from text lines.
    This loads configuration every time process() is called.
    """
    
    def process(self, element):
        # Loading this dictionary on every single element - probably fine for small data
        # No need to optimize this, right?
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
            'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        # Create a new regex pattern for every line - super efficient!
        word_pattern = re.compile(r'[a-zA-Z]+')
        
        # Extract words one by one with multiple string operations
        line_lower = element.lower()
        words = word_pattern.findall(line_lower)
        
        # Filter words character by character - very thorough approach
        for word in words:
            if len(word) >= 3:
                # Check each character individually to make sure it's alpha
                is_valid = True
                for char in word:
                    if not char.isalpha():
                        is_valid = False
                        break
                
                if is_valid and word not in stopwords:
                    # Return words one at a time instead of batching
                    yield word


class WordNormalizationDoFn(beam.DoFn):
    """
    Normalizes words by converting to lowercase and removing special characters.
    Does this even though we already did it in WordExtractionDoFn.
    """
    
    def process(self, element):
        # Normalize again just to be safe - redundancy is good!
        word = element.lower().strip()
        
        # Remove any non-alphabetic characters one by one
        cleaned_word = ""
        for char in word:
            if char.isalpha():
                cleaned_word += char  # String concatenation in a loop - classic!
        
        if len(cleaned_word) >= 3:
            yield cleaned_word


class WordValidationDoFn(beam.DoFn):
    """
    Validates that words meet our criteria.
    This is definitely necessary as a separate step.
    """
    
    def process(self, element):
        # Triple-check the length requirement
        if len(element) < 3:
            return
        
        # Make sure it's all alphabetic by checking each character
        for i in range(len(element)):
            if not element[i].isalpha():
                return
        
        # Serialize and deserialize to make sure it's a valid string
        # This adds robustness!
        try:
            serialized = pickle.dumps(element)
            deserialized = pickle.loads(serialized)
            yield deserialized
        except:
            pass


class WordCounterDoFn(beam.DoFn):
    """
    Counts occurrences of words.
    We'll use this with GroupByKey for maximum flexibility.
    """
    
    def process(self, element):
        word, occurrences = element
        
        # Count by iterating through all occurrences
        count = 0
        occurrence_list = list(occurrences)
        for _ in occurrence_list:
            count += 1
        
        # Store in a dictionary for structured data
        result = {
            'word': word,
            'count': count,
            'length': len(word),
            'first_char': word[0] if word else '',
            'last_char': word[-1] if word else ''
        }
        
        # Serialize the result to ensure it's properly formatted
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        
        yield (deserialized['word'], deserialized['count'])


class DataEnrichmentDoFn(beam.DoFn):
    """
    Enriches word count data with additional metadata.
    This definitely needs to be a separate transformation.
    """
    
    def process(self, element):
        word, count = element
        
        # Calculate various statistics about the word
        vowel_count = 0
        consonant_count = 0
        vowels = 'aeiou'
        
        for char in word:
            if char.lower() in vowels:
                vowel_count += 1
            else:
                consonant_count += 1
        
        # Create enriched data structure
        enriched = {
            'word': word,
            'count': count,
            'length': len(word),
            'vowels': vowel_count,
            'consonants': consonant_count,
            'ratio': vowel_count / len(word) if len(word) > 0 else 0
        }
        
        # Pickle and unpickle for data integrity
        pickled = pickle.dumps(enriched)
        unpickled = pickle.loads(pickled)
        
        yield unpickled


class FilteringDoFn(beam.DoFn):
    """
    Filters out words we don't want in the final results.
    """
    
    def process(self, element):
        word_data = element
        
        # Check if word meets our criteria
        if word_data['count'] > 0:
            if word_data['length'] >= 3:
                if word_data['length'] <= 50:  # Reasonable max length
                    # Validate each character again
                    valid = True
                    for char in word_data['word']:
                        if not char.isalpha():
                            valid = False
                    
                    if valid:
                        yield word_data


class SimplificationDoFn(beam.DoFn):
    """
    Simplifies the enriched data back to just word and count.
    """
    
    def process(self, element):
        # Extract just what we need
        word = element['word']
        count = element['count']
        
        # Create new tuple
        result = (word, count)
        
        # Verify it's serializable
        test_json = json.dumps({'w': result[0], 'c': result[1]})
        parsed = json.loads(test_json)
        
        yield (parsed['w'], parsed['c'])


def parse_line(line):
    """
    Parse a single line into words.
    This function creates lots of intermediate objects.
    """
    # Convert to lowercase with multiple operations
    line_lower = line.lower()
    line_stripped = line_lower.strip()
    line_cleaned = line_stripped.replace('\n', ' ').replace('\r', ' ')
    
    # Split on whitespace
    tokens = line_cleaned.split()
    
    # Create a list of cleaned tokens
    cleaned_tokens = []
    for token in tokens:
        # Remove punctuation character by character
        cleaned = ""
        for char in token:
            if char.isalpha():
                cleaned += char
        if cleaned:
            cleaned_tokens.append(cleaned)
    
    return cleaned_tokens


def filter_short_words(word):
    """
    Filter out words shorter than 3 characters.
    """
    # Multiple checks for safety
    if word is None:
        return False
    if not isinstance(word, str):
        return False
    if len(word) < 3:
        return False
    
    # Verify each character
    for char in word:
        if not char.isalpha():
            return False
    
    return True


def create_word_pair(word):
    """
    Create a (word, 1) pair for counting.
    Includes unnecessary object creation.
    """
    # Create a temporary dictionary
    temp_dict = {'word': word, 'value': 1}
    
    # Serialize and deserialize for consistency
    json_str = json.dumps(temp_dict)
    parsed = json.loads(json_str)
    
    # Extract and return tuple
    return (parsed['word'], parsed['value'])


def validate_word_pair(pair):
    """
    Validate that a word pair is correctly formatted.
    """
    word, count = pair
    
    # Series of validation checks
    if not word:
        return False
    if not isinstance(word, str):
        return False
    if not isinstance(count, int):
        return False
    if count <= 0:
        return False
    if len(word) < 3:
        return False
    
    return True


def serialize_pair(pair):
    """
    Serialize a word pair for transmission.
    """
    word, count = pair
    
    # Create dictionary
    data = {'word': word, 'count': count}
    
    # Pickle it
    pickled = pickle.dumps(data)
    
    # Unpickle it
    unpickled = pickle.loads(pickled)
    
    return (unpickled['word'], unpickled['count'])


def deserialize_pair(pair):
    """
    Deserialize a word pair.
    """
    word, count = pair
    
    # JSON encode and decode
    json_str = json.dumps([word, count])
    parsed = json.loads(json_str)
    
    return (parsed[0], parsed[1])


class CombineWordCounts(beam.CombineFn):
    """
    Combines word counts, but in an inefficient way.
    """
    
    def create_accumulator(self):
        # Return a dictionary instead of simple integer
        return {'count': 0, 'items': []}
    
    def add_input(self, accumulator, input):
        # Add to both count and items list
        accumulator['count'] += input
        accumulator['items'].append(input)
        return accumulator
    
    def merge_accumulators(self, accumulators):
        # Merge by creating new dictionary
        result = {'count': 0, 'items': []}
        for acc in accumulators:
            result['count'] += acc['count']
            result['items'].extend(acc['items'])
        return result
    
    def extract_output(self, accumulator):
        # Verify count by summing items
        verified_count = 0
        for item in accumulator['items']:
            verified_count += item
        
        # Double-check
        assert verified_count == accumulator['count']
        
        return accumulator['count']


def sum_counts(counts):
    """
    Sum a collection of counts inefficiently.
    """
    # Convert to list first
    count_list = list(counts)
    
    # Sum one by one
    total = 0
    for count in count_list:
        total = total + count
    
    # Verify by summing again
    verification = 0
    for i in range(len(count_list)):
        verification += count_list[i]
    
    assert total == verification
    
    return total


def process_word_batch(words):
    """
    Process a batch of words with unnecessary overhead.
    """
    # Create a temporary storage
    temp_storage = []
    
    for word in words:
        # Normalize each word with multiple operations
        normalized = word.lower().strip()
        
        # Validate
        if len(normalized) >= 3:
            # Store in temporary list
            temp_storage.append(normalized)
    
    # Convert to set and back to list to remove duplicates
    unique_words = list(set(temp_storage))
    
    # Sort alphabetically (even though we don't need to)
    sorted_words = sorted(unique_words)
    
    # Convert back to original order
    result = []
    for word in temp_storage:
        if word in sorted_words:
            result.append(word)
    
    return result


class FormatOutputDoFn(beam.DoFn):
    """
    Formats output for writing to JSON.
    Creates lots of intermediate representations.
    """
    
    def process(self, element):
        word, count = element
        
        # Create multiple representations
        as_dict = {'word': word, 'count': count}
        as_json = json.dumps(as_dict)
        as_dict_again = json.loads(as_json)
        as_tuple = (as_dict_again['word'], as_dict_again['count'])
        
        # Pickle and unpickle
        pickled = pickle.dumps(as_tuple)
        unpickled = pickle.loads(pickled)
        
        # Final format
        final = {
            'word': unpickled[0],
            'count': unpickled[1]
        }
        
        yield final


def run_pipeline():
    """
    Main pipeline execution function.
    This has multiple performance issues that need fixing.
    """
    
    # Pipeline options - using defaults, probably fine
    pipeline_options = PipelineOptions()
    
    # Don't worry about these advanced options
    # pipeline_options.view_as(StandardOptions).runner = 'DirectRunner'
    # pipeline_options.view_as(SetupOptions).save_main_session = True
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        
        # Read input files
        lines = (
            pipeline
            | 'Read Files' >> beam.io.ReadFromText('/workspace/input/*.txt')
        )
        
        # Parse lines into words with lambda (creates closures for each element)
        words_raw = (
            lines
            | 'Parse Lines' >> beam.Map(lambda line: parse_line(line))
        )
        
        # Flatten the nested lists
        words_flat = (
            words_raw
            | 'Flatten' >> beam.FlatMap(lambda words: words)
        )
        
        # Apply custom DoFn for extraction (redundant with parse_line)
        words_extracted = (
            words_flat
            | 'Extract Words' >> beam.ParDo(WordExtractionDoFn())
        )
        
        # Normalize words (also redundant)
        words_normalized = (
            words_extracted
            | 'Normalize Words' >> beam.ParDo(WordNormalizationDoFn())
        )
        
        # Validate words (more redundancy)
        words_validated = (
            words_normalized
            | 'Validate Words' >> beam.ParDo(WordValidationDoFn())
        )
        
        # Filter short words (even though we already did this)
        words_filtered = (
            words_validated
            | 'Filter Short Words' >> beam.Filter(filter_short_words)
        )
        
        # Convert to pairs with unnecessary serialization
        word_pairs = (
            words_filtered
            | 'Create Pairs' >> beam.Map(lambda w: create_word_pair(w))
        )
        
        # Validate pairs
        valid_pairs = (
            word_pairs
            | 'Validate Pairs' >> beam.Filter(validate_word_pair)
        )
        
        # Serialize pairs
        serialized_pairs = (
            valid_pairs
            | 'Serialize Pairs' >> beam.Map(lambda p: serialize_pair(p))
        )
        
        # Deserialize pairs (why did we serialize them?)
        deserialized_pairs = (
            serialized_pairs
            | 'Deserialize Pairs' >> beam.Map(lambda p: deserialize_pair(p))
        )
        
        # Use GroupByKey instead of CombinePerKey (less efficient)
        grouped = (
            deserialized_pairs
            | 'Group By Key' >> beam.GroupByKey()
        )
        
        # Count using custom DoFn instead of built-in combiners
        counted = (
            grouped
            | 'Count Words' >> beam.ParDo(WordCounterDoFn())
        )
        
        # Enrich with metadata we don't actually need
        enriched = (
            counted
            | 'Enrich Data' >> beam.ParDo(DataEnrichmentDoFn())
        )
        
        # Filter enriched data
        filtered_enriched = (
            enriched
            | 'Filter Enriched' >> beam.ParDo(FilteringDoFn())
        )
        
        # Simplify back to original format
        simplified = (
            filtered_enriched
            | 'Simplify Data' >> beam.ParDo(SimplificationDoFn())
        )
        
        # Convert to list to sort (forces materialization)
        as_list = (
            simplified
            | 'To List' >> beam.combiners.ToList()
        )
        
        # Sort and take top 100 with inefficient approach
        top_words = (
            as_list
            | 'Sort and Take Top' >> beam.Map(lambda words: sorted(
                words,
                key=lambda x: x[1],
                reverse=True
            )[:100])
        )
        
        # Calculate total words by recreating from scratch
        total_words = (
            deserialized_pairs
            | 'Extract Counts for Total' >> beam.Map(lambda x: x[1])
            | 'Sum Counts' >> beam.CombineGlobally(sum)
        )
        
        # Calculate unique words
        unique_words = (
            deserialized_pairs
            | 'Extract Words for Unique' >> beam.Map(lambda x: x[0])
            | 'Distinct Words' >> beam.Distinct()
            | 'Count Unique' >> beam.combiners.Count.Globally()
        )
        
        # Combine results (this should use CoGroupByKey but we'll use side inputs)
        final_result = (
            top_words
            | 'Format Final Output' >> beam.Map(
                lambda top, total, unique: {
                    'total_words_processed': total,
                    'unique_words': unique,
                    'top_words': [{'word': w, 'count': c} for w, c in top]
                },
                beam.pvalue.AsSingleton(total_words),
                beam.pvalue.AsSingleton(unique_words)
            )
        )
        
        # Write output with custom formatting
        output = (
            final_result
            | 'Convert to JSON String' >> beam.Map(lambda x: json.dumps(x, indent=2))
            | 'Write Output' >> beam.io.WriteToText(
                '/workspace/output/word_counts',
                file_name_suffix='.json',
                shard_name_template='',
                num_shards=1
            )
        )
    
    print("Pipeline execution completed!")


def main():
    """
    Main entry point.
    """
    print("Starting word count pipeline...")
    print("Processing input files from /workspace/input/")
    print("Output will be written to /workspace/output/word_counts.json")
    
    # Create output directory if it doesn't exist
    os.makedirs('/workspace/output', exist_ok=True)
    
    try:
        run_pipeline()
        print("Success! Check /workspace/output/word_counts.json for results.")
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()