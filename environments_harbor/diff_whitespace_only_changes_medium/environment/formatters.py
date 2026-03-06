#!/usr/bin/env python3

import json
import xml.etree.ElementTree as ET
from datetime import datetime


def format_date(date_str):
    """Format a date string to a standard format."""
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d, %Y')
    except ValueError:
        return date_str


def format_currency(amount):
    """Format a number as currency with dollar sign."""
    try:
        return f"${float(amount):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"



def format_phone(phone):
    """Format a phone number to (XXX) XXX-XXXX format."""
    digits = ''.join(filter(str.isdigit, str(phone)))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone


class Formatter:
    """A class for formatting data in various formats."""
    
    def __init__(self):
        self.indent = 4
    
    def format_json(self, data):
        """Format data as JSON string."""
        try:
            return json.dumps(data, indent=self.indent, sort_keys=True)
        except (TypeError, ValueError) as e:
            return f"Error formatting JSON: {e}"
    
    
    def format_xml(self, data):
        """Format data as XML string."""
        try:
            root = ET.Element('root')
            self._dict_to_xml(data, root)
            return ET.tostring(root, encoding='unicode')
        except Exception as e:
            return f"Error formatting XML: {e}"
    
    def _dict_to_xml(self, data, parent):
        """Helper method to convert dictionary to XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                element = ET.SubElement(parent, str(key))
                if isinstance(value, (dict, list)):
                    self._dict_to_xml(value, element)
                else:
                    element.text = str(value)
        elif isinstance(data, list):
            for item in data:
                element = ET.SubElement(parent, 'item')
                if isinstance(item, (dict, list)):
                    self._dict_to_xml(item, element)
                else:
                    element.text = str(item)