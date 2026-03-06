#!/usr/bin/env python3

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import re

class ExtractWords(beam.DoFn):
    def process(self, element):
        # Bug 1: Not converting to lowercase here
        words = re.findall(r'\w+', element)
        return words

class FormatOutput(beam.DoFn):
    def process(self, element):
        word, count = element
        # Bug 2: Wrong format - missing space after colon
        return [f"{word}:{count}"]

def run_pipeline():
    # Bug 3: Missing pipeline options setup
    pipeline_options = None
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read input files
        lines = pipeline | 'Read Files' >> beam.io.ReadFromText(
            '/workspace/input_data/*.txt'
        )
        
        # Extract words
        # Bug 4: Using Map instead of FlatMap - will create nested lists
        words = lines | 'Extract Words' >> beam.Map(ExtractWords())
        
        # Count word frequencies
        word_counts = (
            words
            | 'Pair With One' >> beam.Map(lambda word: (word, 1))
            | 'Group and Sum' >> beam.CombinePerKey(sum)
        )
        
        # Bug 5: Incorrect filter logic (using > instead of >=)
        filtered_counts = (
            word_counts
            | 'Filter Rare Words' >> beam.Filter(lambda wc: wc[1] > 3)
        )
        
        # Sort by frequency (descending)
        sorted_counts = (
            filtered_counts
            | 'Convert to List' >> beam.combiners.ToList()
            | 'Sort' >> beam.Map(lambda lst: sorted(lst, key=lambda x: x[1], reverse=True))
            | 'Flatten' >> beam.FlatMap(lambda x: x)
        )
        
        # Format output
        formatted = (
            sorted_counts
            | 'Format Output' >> beam.ParDo(FormatOutput())
        )
        
        # Write results
        formatted | 'Write Results' >> beam.io.WriteToText(
            '/workspace/word_frequencies.txt',
            shard_name_template='',
            header=''
        )

if __name__ == '__main__':
    run_pipeline()