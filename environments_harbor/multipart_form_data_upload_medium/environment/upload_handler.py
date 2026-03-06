#!/usr/bin/env python3

import sys
import json
import re

def parse_multipart(file_path):
    """
    Attempt to parse multipart form data
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    
    # Convert to string for easier parsing
    data = raw_data.decode('utf-8', errors='ignore')
    
    # Try to find the boundary - BUG: not handling the boundary correctly
    # Should look for it in the first line, but missing the -- prefix handling
    first_line = data.split('\n')[0]
    boundary = first_line.strip()  # BUG: includes the -- prefix which causes issues
    
    # Split by boundary - BUG: this will fail because boundary format is wrong
    parts = data.split(boundary)
    
    filename = None
    content_type = None
    file_content = None
    
    # Look for the file part
    for part in parts:
        if 'Content-Disposition' in part:
            # Try to extract filename - BUG: regex is too simple and doesn't handle quotes properly
            filename_match = re.search(r'filename=(\S+)', part)  # BUG: doesn't handle quoted filenames
            if filename_match:
                filename = filename_match.group(1)
                # BUG: filename might have quotes or semicolons attached
            
            # Try to extract content type
            content_type_match = re.search(r'Content-Type:\s*(\S+)', part)
            if content_type_match:
                content_type = content_type_match.group(1)
            
            # Try to extract file content - BUG: doesn't properly skip headers
            # Should split on \r\n\r\n or \n\n, but this is too naive
            lines = part.split('\n')
            # BUG: just taking everything after line 3, but headers might vary
            # and this includes extra newlines and boundary markers
            if len(lines) > 3:
                file_content = '\n'.join(lines[3:])  # BUG: wrong join, corrupts binary data
    
    # Calculate file size - BUG: using string length instead of binary length
    # and file_content includes boundary markers and extra data
    file_size = len(file_content) if file_content else 0
    
    # Output results
    result = {
        'filename': filename,
        'content_type': content_type,
        'file_size': file_size
    }
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python upload_handler.py <multipart_file>")
        sys.exit(1)
    
    parse_multipart(sys.argv[1])