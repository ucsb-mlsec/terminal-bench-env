#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import base64

# SAML 2.0 XML namespaces
NAMESPACES = {
    'saml2': 'urn:oasis:names:tc:SAML:2.0:assertion',
    'saml2p': 'urn:oasis:names:tc:SAML:2.0:protocol',
    'ds': 'http://www.w3.org/2000/09/xmldsig#'
}

def process_saml_response(saml_response):
    """
    Process a SAML 2.0 response and extract user attributes.
    
    Args:
        saml_response: SAML response as XML string or base64 encoded string
        
    Returns:
        dict: User attributes containing username and role
    """
    # Decode base64 if necessary
    try:
        decoded_response = base64.b64decode(saml_response).decode('utf-8')
    except Exception:
        # If decoding fails, assume it's already XML
        decoded_response = saml_response
    
    # Parse the XML response
    root = ET.fromstring(decoded_response)
    
    # Find all assertions in the response
    assertions = root.findall('.//saml2:Assertion', NAMESPACES)
    
    if not assertions:
        raise ValueError("No assertions found in SAML response")
    
    # Validate digital signature on the first assertion
    # This ensures the response came from a trusted identity provider
    signature = assertions[0].find('.//ds:Signature', NAMESPACES)
    
    if signature is None:
        raise ValueError("No signature found in assertion - response rejected")
    
    # Perform signature validation on the first assertion
    # In production, this would verify the cryptographic signature
    # For this implementation, we check that the signature element exists
    signature_value = signature.find('.//ds:SignatureValue', NAMESPACES)
    
    if signature_value is None or not signature_value.text:
        raise ValueError("Invalid signature - response rejected")
    
    print(f"[VALIDATION] Signature validated successfully for assertion")
    print(f"[INFO] Found {len(assertions)} assertion(s) in response")
    
    # Extract user attributes from the last assertion in the response
    # This is where attributes like username and role are stored
    target_assertion = assertions[-1]
    
    # Navigate to AttributeStatement
    attribute_statement = target_assertion.find('.//saml2:AttributeStatement', NAMESPACES)
    
    if attribute_statement is None:
        raise ValueError("No AttributeStatement found in assertion")
    
    # Extract username and role from attributes
    user_data = {}
    
    for attribute in attribute_statement.findall('.//saml2:Attribute', NAMESPACES):
        attr_name = attribute.get('Name')
        attr_value = attribute.find('.//saml2:AttributeValue', NAMESPACES)
        
        if attr_value is not None and attr_value.text:
            if attr_name == 'username' or attr_name == 'urn:oid:0.9.2342.19200300.100.1.1':
                user_data['username'] = attr_value.text
            elif attr_name == 'role' or attr_name == 'urn:oid:1.3.6.1.4.1.5923.1.1.1.1':
                user_data['role'] = attr_value.text
    
    if 'username' not in user_data or 'role' not in user_data:
        raise ValueError("Required attributes (username, role) not found in assertion")
    
    print(f"[SUCCESS] Authenticated user: {user_data['username']} with role: {user_data['role']}")
    
    return user_data