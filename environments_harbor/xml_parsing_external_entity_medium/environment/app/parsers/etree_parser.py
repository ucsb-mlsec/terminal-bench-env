#!/usr/bin/env python3
"""
XML Parser using xml.etree.ElementTree
This module parses XML documents and extracts data.
WARNING: This version is vulnerable to XXE attacks!
"""

import xml.etree.ElementTree as ET


def parse_xml(xml_string):
    """
    Parse an XML string and extract data from it.
    
    Args:
        xml_string: String containing XML data
        
    Returns:
        Dictionary containing parsed data
    """
    # Parse the XML string - VULNERABLE: no protection against XXE
    root = ET.fromstring(xml_string)
    
    # Extract data from the XML structure
    result = {
        'root_tag': root.tag,
        'items': []
    }
    
    # Process all child elements
    for child in root:
        item_data = {
            'tag': child.tag,
            'text': child.text if child.text else '',
            'attributes': child.attrib
        }
        result['items'].append(item_data)
    
    return result


def parse_xml_file(file_path):
    """
    Parse an XML file and extract data.
    
    Args:
        file_path: Path to XML file
        
    Returns:
        Dictionary containing parsed data
    """
    # Parse XML file - VULNERABLE: no protection against XXE
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    result = {
        'root_tag': root.tag,
        'items': []
    }
    
    for child in root:
        item_data = {
            'tag': child.tag,
            'text': child.text if child.text else '',
            'attributes': child.attrib
        }
        result['items'].append(item_data)
    
    return result