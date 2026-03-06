#!/usr/bin/env python3

# Global list to store processed messages
processed_messages = []
filtered_messages = []

def message_callback(severity, message):
    """
    Callback function for processing debug messages from the graphics rendering pipeline.
    
    Args:
        severity (str): The severity level ('CRITICAL', 'WARNING', 'INFO', 'DEBUG')
        message (str): The debug message content
    """
    # Filter out INFO and DEBUG messages - only process CRITICAL and WARNING
    if severity not in ['CRITICAL', 'WARNING']:
        filtered_messages.append({'severity': severity, 'message': message})
        return
    
    # Process CRITICAL and WARNING messages
    processed_messages.append({'severity': severity, 'message': message})
    
    # Log the message
    print(f"[{severity}] {message}")
    
    # Additional processing for critical messages
    if severity == 'CRITICAL':
        print(f"  --> CRITICAL ERROR DETECTED: Immediate attention required!")

def get_statistics():
    """
    Returns statistics about processed and filtered messages.
    
    Returns:
        dict: Dictionary containing processed and filtered counts
    """
    return {
        'processed_count': len(processed_messages),
        'filtered_count': len(filtered_messages)
    }

def reset_statistics():
    """
    Resets the message counters for testing purposes.
    """
    global processed_messages, filtered_messages
    processed_messages = []
    filtered_messages = []