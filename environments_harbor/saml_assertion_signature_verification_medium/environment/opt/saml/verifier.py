#!/usr/bin/env python3

import sys
import os
from lxml import etree
from signxml import XMLVerifier
from cryptography import x509
from cryptography.hazmat.backends import default_backend

def load_certificate(cert_path):
    """Load X.509 certificate from file"""
    with open(cert_path, 'rb') as f:
        cert_data = f.read()
    cert = x509.load_pem_x509_certificate(cert_data, default_backend())
    return cert

def load_assertion(assertion_path):
    """Load SAML assertion XML from file"""
    with open(assertion_path, 'rb') as f:
        xml_data = f.read()
    return xml_data

def verify_saml_assertion(assertion_xml, certificate):
    """Verify the digital signature on a SAML assertion"""
    try:
        # Parse the XML assertion
        root = etree.fromstring(assertion_xml)
        
        # Define SAML namespace
        namespaces = {
            'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
            'ds': 'http://www.w3.org/2000/09/xmldsig#'
        }
        
        # Find the Signature element
        signature = root.find('.//ds:Signature', namespaces=namespaces)
        
        if signature is None:
            print("Error: No signature found in assertion")
            return False
        
        # Extract certificate public key in PEM format
        cert_pem = certificate.public_bytes(encoding=x509.Encoding.PEM)
        
        # Verify the signature - BUG: using the wrong element for verification
        # Should verify the entire assertion, but instead verifying just the signature element
        verifier = XMLVerifier()
        verified_data = verifier.verify(
            data=etree.tostring(signature),  # BUG: should be 'root' not 'signature'
            x509_cert=cert_pem
        )
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: verifier.py <assertion_xml_file>")
        sys.exit(1)
    
    assertion_path = sys.argv[1]
    cert_path = "/opt/saml/idp_cert.pem"
    
    if not os.path.exists(assertion_path):
        print(f"Error: Assertion file not found: {assertion_path}")
        sys.exit(1)
    
    if not os.path.exists(cert_path):
        print(f"Error: Certificate file not found: {cert_path}")
        sys.exit(1)
    
    # Load certificate and assertion
    cert = load_certificate(cert_path)
    assertion_xml = load_assertion(assertion_path)
    
    # Verify the assertion
    if verify_saml_assertion(assertion_xml, cert):
        print("Assertion verification successful")
        sys.exit(0)
    else:
        print("Assertion verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()