#!/usr/bin/env python3
"""
API Handler Module - Parses XML payloads from incoming API requests
WARNING: This module is vulnerable to XXE attacks
"""

from lxml import etree
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_api_request(xml_payload):
    """
    Parse XML payload from API request and extract request parameters.
    
    Args:
        xml_payload: XML data as bytes or string
        
    Returns:
        dict: Parsed request data with parameters
        
    Raises:
        Exception: If XML parsing fails
    """
    try:
        # Convert string to bytes if necessary
        if isinstance(xml_payload, str):
            xml_payload = xml_payload.encode('utf-8')
        
        # Create vulnerable XML parser - does NOT disable entity resolution
        parser = etree.XMLParser()
        
        # Parse the XML payload - VULNERABLE to XXE
        root = etree.fromstring(xml_payload, parser=parser)
        
        # Extract request parameters from XML structure
        request_data = {
            'action': root.get('action', 'unknown'),
            'timestamp': root.get('timestamp', ''),
            'parameters': {}
        }
        
        # Extract parameters from child elements
        for param in root.findall('.//parameter'):
            param_name = param.get('name')
            param_value = param.text
            if param_name:
                request_data['parameters'][param_name] = param_value
        
        # Extract any data elements
        data_elements = root.findall('.//data')
        if data_elements:
            request_data['data'] = [elem.text for elem in data_elements if elem.text]
        
        logger.info(f"Successfully parsed API request with action: {request_data['action']}")
        return request_data
        
    except etree.XMLSyntaxError as e:
        logger.error(f"XML syntax error: {e}")
        raise Exception(f"Invalid XML format: {e}")
    except Exception as e:
        logger.error(f"Error parsing API request: {e}")
        raise


def process_xml_api_call(xml_data):
    """
    Process XML API call and execute the requested command.
    
    Args:
        xml_data: XML data as bytes or string containing API command
        
    Returns:
        dict: Response data with status and results
        
    Raises:
        Exception: If processing fails
    """
    try:
        # Parse the XML data using vulnerable parser
        if isinstance(xml_data, str):
            xml_data = xml_data.encode('utf-8')
        
        # Use vulnerable XMLParser without security restrictions
        parser = etree.XMLParser()
        root = etree.fromstring(xml_data, parser=parser)
        
        # Extract command and parameters
        command = root.find('.//command')
        if command is None:
            raise Exception("No command found in XML")
        
        cmd_type = command.get('type', 'unknown')
        cmd_data = command.text or ''
        
        # Process different API commands
        response = {
            'status': 'success',
            'command': cmd_type,
            'timestamp': root.get('timestamp', ''),
            'result': None
        }
        
        if cmd_type == 'query':
            # Process query command
            query_params = {}
            for param in root.findall('.//param'):
                key = param.get('key')
                value = param.text
                if key:
                    query_params[key] = value
            
            response['result'] = {
                'query': cmd_data,
                'parameters': query_params,
                'records': []
            }
            logger.info(f"Processed query command: {cmd_data}")
            
        elif cmd_type == 'update':
            # Process update command
            update_fields = {}
            for field in root.findall('.//field'):
                field_name = field.get('name')
                field_value = field.text
                if field_name:
                    update_fields[field_name] = field_value
            
            response['result'] = {
                'updated': True,
                'fields': update_fields,
                'affected_rows': len(update_fields)
            }
            logger.info(f"Processed update command with {len(update_fields)} fields")
            
        elif cmd_type == 'delete':
            # Process delete command
            delete_id = root.find('.//id')
            delete_id_value = delete_id.text if delete_id is not None else None
            
            response['result'] = {
                'deleted': True,
                'id': delete_id_value
            }
            logger.info(f"Processed delete command for id: {delete_id_value}")
            
        else:
            response['status'] = 'error'
            response['message'] = f"Unknown command type: {cmd_type}"
            logger.warning(f"Unknown command type: {cmd_type}")
        
        return response
        
    except etree.XMLSyntaxError as e:
        logger.error(f"XML syntax error in API call: {e}")
        return {
            'status': 'error',
            'message': f"Invalid XML format: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error processing API call: {e}")
        return {
            'status': 'error',
            'message': f"Processing failed: {str(e)}"
        }


def validate_api_request(xml_payload):
    """
    Validate that API request XML has required structure.
    
    Args:
        xml_payload: XML data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        parsed = parse_api_request(xml_payload)
        return 'action' in parsed and parsed['action'] != 'unknown'
    except Exception:
        return False