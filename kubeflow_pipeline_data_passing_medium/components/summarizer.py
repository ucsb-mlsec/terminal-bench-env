import json
import os

def summarizer_component(input_data_path: str) -> dict:
    """
    Calculate statistics from filtered sensor readings.
    
    Args:
        input_data_path: Path to JSON file containing filtered readings
        
    Returns:
        Dictionary containing calculated statistics
    """
    # Read the filtered data from the input file path
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    
    readings = data.get('readings', [])
    
    if not readings:
        result = {
            'average_temperature': 0.0,
            'count': 0,
            'message': 'No readings to process'
        }
    else:
        # Calculate average temperature
        total_temp = sum(r['temperature'] for r in readings)
        avg_temp = total_temp / len(readings)
        
        result = {
            'average_temperature': round(avg_temp, 2),
            'count': len(readings),
            'min_temperature': min(r['temperature'] for r in readings),
            'max_temperature': max(r['temperature'] for r in readings)
        }
    
    # Save result to output file
    output_path = '/tmp/summary_output.json'
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Summary calculated: {result}")
    
    return result