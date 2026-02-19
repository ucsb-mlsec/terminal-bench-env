#!/usr/bin/env python3

import xml.dom.minidom

def parse_xml(xml_string):
    """
    Parse XML string using minidom and extract data.
    
    Args:
        xml_string: XML content as string
        
    Returns:
        Dictionary containing parsed data
    """
    # Parse XML string - VULNERABLE: uses default settings
    doc = xml.dom.minidom.parseString(xml_string)
    
    # Extract data from XML
    result = {
        'items': [],
        'attributes': {}
    }
    
    # Get all item elements
    items = doc.getElementsByTagName('item')
    for item in items:
        if item.firstChild:
            result['items'].append(item.firstChild.nodeValue)
    
    # Get all data elements
    data_elements = doc.getElementsByTagName('data')
    for data in data_elements:
        if data.hasAttribute('name'):
            name = data.getAttribute('name')
            if data.firstChild:
                result['attributes'][name] = data.firstChild.nodeValue
    
    return result


def parse_xml_file(filepath):
    """
    Parse XML file using minidom.
    
    Args:
        filepath: Path to XML file
        
    Returns:
        Dictionary containing parsed data
    """
    with open(filepath, 'r') as f:
        xml_content = f.read()
    return parse_xml(xml_content)