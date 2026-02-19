import json

def processor_component(input_data):
    # Broken: trying to use input_data directly as a dict instead of reading from file
    data = input_data
    
    # Filter readings between 20 and 30 degrees
    readings = [r for r in data['readings'] if 20 <= r['temperature'] <= 30]
    
    filtered_data = {"readings": readings}
    
    # Broken: not saving to file or returning a path
    return filtered_data