#!/usr/bin/env python3

from scicore import math_utils, data_ops
from dataproc.pipeline import ProcessingPipeline
from dataproc import transformers

def run_analysis():
    # Create sample data
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Create and execute pipeline
    pipeline = ProcessingPipeline(data)
    pipeline.execute()
    
    # Use transformers to scale data
    transformed = transformers.scale_data(data)
    
    # Compute checksum
    checksum = int(sum(transformed))
    
    # Print result
    print(f'Analysis complete: checksum={checksum}')
    
    return checksum

if __name__ == '__main__':
    run_analysis()