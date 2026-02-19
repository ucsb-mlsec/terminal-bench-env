import json
import os

def reader_component(input_file: str):
    """Reads sensor data from input file but doesn't properly return or save it."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print('Data loaded')
    # Bug: Not returning data or writing to output file