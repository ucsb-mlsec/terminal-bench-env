#!/usr/bin/env python3
"""
lxml-based XML parser module
This module uses lxml to parse XML documents
"""

from lxml import etree


def parse_xml(xml_string):
    """
    Parse XML string and extract data
    
    Args:
        xml_string: XML content as string
        
    Returns:
        Dictionary containing parsed data
    """
    # Parse XML with default settings (VULNERABLE - no security restrictions)
    root = etree.fromstring(xml_string.encode('utf-8'))
    
    # Extract data from XML elements
    result = {}
    
    # Handle simple data structure
    if root.tag == 'data':
        items = []
        for item in root.findall('.//item'):
            items.append(item.text if item.text else '')
        result['items'] = items
    
    # Generic extraction for any structure
    result['root_tag'] = root.tag
    result['root_text'] = root.text if root.text else ''
    
    # Extract all child elements
    children = {}
    for child in root:
        children[child.tag] = child.text if child.text else ''
    result['children'] = children
    
    return result