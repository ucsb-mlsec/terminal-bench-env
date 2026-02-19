#!/usr/bin/env python3
"""
Data Importer Module - Bulk XML Data Import Service

This module handles bulk data imports from user-uploaded XML files.
It processes XML documents containing multiple data records and converts
them into Python data structures for further processing.

WARNING: This module is vulnerable to XXE attacks due to unsafe XML parsing.
"""

import xml.etree.ElementTree as ET
import os
from typing import List, Dict, Any


def import_data_from_xml(xml_file_path: str) -> List[Dict[str, Any]]:
    """
    Import data records from an XML file.
    
    This function reads an XML file and extracts data records from it.
    Expected XML structure:
    <data>
        <record>
            <id>1</id>
            <name>Example</name>
            <value>Some value</value>
        </record>
        ...
    </data>
    
    Args:
        xml_file_path: Path to the XML file to process
        
    Returns:
        List of dictionaries, each containing a data record
        
    Raises:
        FileNotFoundError: If the XML file doesn't exist
        ET.ParseError: If the XML is malformed
    """
    records = []
    
    try:
        # Parse the XML file - VULNERABLE: external entities are enabled by default
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Extract all record elements
        for record_elem in root.findall('.//record'):
            record_data = {}
            
            # Extract child elements from each record
            for child in record_elem:
                # Get the tag name and text content
                tag_name = child.tag
                tag_value = child.text if child.text else ""
                record_data[tag_name] = tag_value
            
            # Only add records that have data
            if record_data:
                records.append(record_data)
        
        return records
        
    except FileNotFoundError:
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing XML file: {str(e)}")


def validate_and_import(xml_content_string: str) -> List[Dict[str, Any]]:
    """
    Validate and import data from XML content string.
    
    This function takes raw XML content as a string, validates its structure,
    and extracts data records. Useful for processing XML data received from
    API requests or other sources where the data isn't in a file.
    
    Args:
        xml_content_string: XML content as a string
        
    Returns:
        List of dictionaries containing parsed data records
        
    Raises:
        ValueError: If XML content is empty or invalid
        ET.ParseError: If XML parsing fails
    """
    if not xml_content_string or not xml_content_string.strip():
        raise ValueError("XML content cannot be empty")
    
    records = []
    
    try:
        # Parse XML from string - VULNERABLE: no protection against XXE
        root = ET.fromstring(xml_content_string)
        
        # Process records from the parsed XML
        for record_elem in root.findall('.//record'):
            record_data = {}
            
            # Extract all child elements
            for child in record_elem:
                tag_name = child.tag
                tag_value = child.text if child.text else ""
                record_data[tag_name] = tag_value
            
            if record_data:
                records.append(record_data)
        
        return records
        
    except ET.ParseError as e:
        raise ET.ParseError(f"Invalid XML format: {str(e)}")
    except Exception as e:
        raise Exception(f"Error validating XML content: {str(e)}")


def get_import_statistics(xml_file_path: str) -> Dict[str, int]:
    """
    Get statistics about the data in an XML file without fully importing it.
    
    Args:
        xml_file_path: Path to the XML file
        
    Returns:
        Dictionary with statistics (record_count, field_count, etc.)
    """
    try:
        # Parse to get statistics - also vulnerable
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        records = root.findall('.//record')
        record_count = len(records)
        
        # Count unique field names across all records
        field_names = set()
        for record in records:
            for child in record:
                field_names.add(child.tag)
        
        return {
            'record_count': record_count,
            'unique_fields': len(field_names),
            'field_names': list(field_names)
        }
        
    except Exception as e:
        raise Exception(f"Error getting statistics: {str(e)}")