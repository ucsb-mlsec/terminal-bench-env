#!/usr/bin/env python3

# WARNING: This XML parser is intentionally vulnerable for educational/testing purposes
# DO NOT use this code in production environments

import sys
from lxml import etree

def parse_xml_vulnerable(xml_input):
    """
    Vulnerable XML parser that allows XXE attacks.
    This parser has resolve_entities=True which allows external entity resolution.
    """
    try:
        # VULNERABLE: This parser configuration allows external entity resolution
        parser = etree.XMLParser(
            resolve_entities=True,  # This enables XXE vulnerability
            no_network=False,       # Allows network access for entities
            dtd_validation=False,
            load_dtd=True          # Loads DTD which can contain entities
        )
        
        # Parse the XML input with the vulnerable parser
        if isinstance(xml_input, str):
            xml_input = xml_input.encode('utf-8')
        
        root = etree.fromstring(xml_input, parser=parser)
        
        # Extract and display parsed content
        result = []
        result.append(f"Root tag: {root.tag}")
        
        # Recursively display all text content
        for elem in root.iter():
            if elem.text and elem.text.strip():
                result.append(f"{elem.tag}: {elem.text}")
        
        output = "\n".join(result)
        print(output)
        return output
        
    except etree.XMLSyntaxError as e:
        print(f"XML Syntax Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing XML: {e}", file=sys.stderr)
        return None

def main():
    """
    Main function to demonstrate the vulnerable XML parser.
    Reads XML from stdin or from a file specified as argument.
    """
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            xml_data = f.read()
    else:
        # Read from stdin
        print("Enter XML (press Ctrl+D when done):", file=sys.stderr)
        xml_data = sys.stdin.read()
    
    if xml_data:
        parse_xml_vulnerable(xml_data)
    else:
        print("No XML input provided", file=sys.stderr)

if __name__ == "__main__":
    main()