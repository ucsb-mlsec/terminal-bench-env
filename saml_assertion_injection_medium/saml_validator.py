#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import sys

# SAML namespace constants
SAML_NS = 'urn:oasis:names:tc:SAML:2.0:assertion'
SAMLP_NS = 'urn:oasis:names:tc:SAML:2.0:protocol'
DS_NS = 'http://www.w3.org/2000/09/xmldsig#'

def validate_saml_response(xml_file_path):
    """
    Validates SAML response with weak signature validation.
    Only checks if Signature element exists, doesn't verify cryptographic validity.
    """
    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Register namespaces for easier searching
        namespaces = {
            'saml': SAML_NS,
            'samlp': SAMLP_NS,
            'ds': DS_NS
        }
        
        # Look for Response element
        response = root if root.tag == f'{{{SAMLP_NS}}}Response' else root.find('.//samlp:Response', namespaces)
        
        if response is None:
            return {'valid': False, 'name_id': None, 'role': None, 'email': None, 'username': None}
        
        # Find Assertion element
        assertion = response.find('.//saml:Assertion', namespaces)
        
        if assertion is None:
            return {'valid': False, 'name_id': None, 'role': None, 'email': None, 'username': None}
        
        # WEAK SIGNATURE VALIDATION - only checks existence, not cryptographic validity
        signature = tree.find('.//{' + DS_NS + '}Signature')
        
        if signature is None:
            return {'valid': False, 'name_id': None, 'role': None, 'email': None, 'username': None}
        
        # Extract NameID
        name_id_element = assertion.find('.//saml:Subject/saml:NameID', namespaces)
        name_id = name_id_element.text if name_id_element is not None else None
        
        # Extract attributes
        role = None
        email = None
        username = None
        
        attribute_statement = assertion.find('.//saml:AttributeStatement', namespaces)
        
        if attribute_statement is not None:
            for attribute in attribute_statement.findall('saml:Attribute', namespaces):
                attr_name = attribute.get('Name')
                attr_value_element = attribute.find('saml:AttributeValue', namespaces)
                attr_value = attr_value_element.text if attr_value_element is not None else None
                
                if attr_name == 'role':
                    role = attr_value
                elif attr_name == 'email':
                    email = attr_value
                elif attr_name == 'username':
                    username = attr_value
        
        return {
            'valid': True,
            'name_id': name_id,
            'role': role,
            'email': email,
            'username': username
        }
        
    except Exception as e:
        print(f"Error parsing SAML response: {e}", file=sys.stderr)
        return {'valid': False, 'name_id': None, 'role': None, 'email': None, 'username': None}

def check_access(validation_result):
    """
    Checks access based on validation result.
    """
    if not validation_result['valid']:
        print('Access denied: Invalid SAML assertion')
        return False
    
    if validation_result['role'] == 'admin':
        print('Access granted: admin')
        return True
    
    if validation_result['role'] == 'user':
        print('Access granted: user')
        return True
    
    print('Access denied: Unknown role')
    return False

if __name__ == '__main__':
    # Accept command line argument for XML file path
    xml_file_path = sys.argv[1] if len(sys.argv) > 1 else '/home/challenge/malicious_saml.xml'
    
    # Validate SAML response
    validation_result = validate_saml_response(xml_file_path)
    
    # Check access
    access_granted = check_access(validation_result)
    
    # Exit with appropriate status code
    sys.exit(0 if access_granted else 1)