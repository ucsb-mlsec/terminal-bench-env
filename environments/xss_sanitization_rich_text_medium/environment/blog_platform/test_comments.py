#!/usr/bin/env python3

import unittest
import json
import os
import sys
import subprocess
import re

class TestCommentSanitizer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Run the sanitizer once before all tests"""
        # Run the sanitizer script
        result = subprocess.run(
            [sys.executable, '/workspace/solution/sanitizer.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Sanitizer failed with error: {result.stderr}")
            raise Exception(f"Sanitizer script failed: {result.stderr}")
        
        # Load sanitized output
        output_path = '/workspace/solution/sanitized_comments.json'
        if not os.path.exists(output_path):
            raise Exception(f"Output file not found: {output_path}")
        
        with open(output_path, 'r') as f:
            sanitized_list = json.load(f)
        
        # Store as dictionary keyed by id
        cls.sanitized_comments = {
            item['id']: item['sanitized_content'] 
            for item in sanitized_list
        }
    
    def assertNoScriptTags(self, content):
        """Helper to check for script tags in any form"""
        content_lower = content.lower()
        self.assertNotIn('<script', content_lower, "Found <script tag")
        self.assertNotIn('</script>', content_lower, "Found </script> tag")
    
    def assertNoJavaScriptURL(self, content):
        """Helper to check for javascript: URLs"""
        content_lower = content.lower()
        # Remove spaces and common obfuscations
        content_clean = re.sub(r'\s+', '', content_lower)
        self.assertNotIn('javascript:', content_clean, "Found javascript: URL")
    
    def assertNoEventHandlers(self, content):
        """Helper to check for event handler attributes"""
        event_handlers = [
            'onclick', 'onload', 'onerror', 'onmouseover', 'onmouseout',
            'onfocus', 'onblur', 'onchange', 'onsubmit', 'onkeydown',
            'onkeyup', 'onkeypress', 'onmousedown', 'onmouseup'
        ]
        content_lower = content.lower()
        for handler in event_handlers:
            self.assertNotIn(handler, content_lower, f"Found event handler: {handler}")
    
    def assertNoDataURI(self, content):
        """Helper to check for data: URIs"""
        content_lower = content.lower()
        if 'data:' in content_lower:
            # data: URIs are dangerous, should be blocked
            self.fail("Found data: URI which can contain scripts")
    
    # XSS Attack Tests (15 tests)
    
    def test_xss_01_basic_script_tag(self):
        """Test blocking of basic script tag"""
        content = self.sanitized_comments.get('xss_01', '')
        self.assertNoScriptTags(content)
    
    def test_xss_02_script_with_alert(self):
        """Test blocking of script tag with alert"""
        content = self.sanitized_comments.get('xss_02', '')
        self.assertNoScriptTags(content)
        self.assertNotIn('alert', content.lower())
    
    def test_xss_03_uppercase_script(self):
        """Test blocking of uppercase/mixed case script tags"""
        content = self.sanitized_comments.get('xss_03', '')
        self.assertNoScriptTags(content)
    
    def test_xss_04_img_onerror(self):
        """Test blocking of img tag with onerror handler"""
        content = self.sanitized_comments.get('xss_04', '')
        self.assertNoEventHandlers(content)
        # img tags should be removed as they're not in allowed list
        self.assertNotIn('<img', content.lower())
    
    def test_xss_05_onclick_handler(self):
        """Test blocking of onclick event handler"""
        content = self.sanitized_comments.get('xss_05', '')
        self.assertNoEventHandlers(content)
    
    def test_xss_06_javascript_url(self):
        """Test blocking of javascript: URL in link"""
        content = self.sanitized_comments.get('xss_06', '')
        self.assertNoJavaScriptURL(content)
    
    def test_xss_07_javascript_url_encoded(self):
        """Test blocking of encoded javascript: URL"""
        content = self.sanitized_comments.get('xss_07', '')
        self.assertNoJavaScriptURL(content)
    
    def test_xss_08_data_uri(self):
        """Test blocking of data: URI with script"""
        content = self.sanitized_comments.get('xss_08', '')
        self.assertNoDataURI(content)
    
    def test_xss_09_svg_onload(self):
        """Test blocking of SVG with onload"""
        content = self.sanitized_comments.get('xss_09', '')
        self.assertNoEventHandlers(content)
        # SVG not in allowed tags
        self.assertNotIn('<svg', content.lower())
    
    def test_xss_10_style_injection(self):
        """Test blocking of style tag with script"""
        content = self.sanitized_comments.get('xss_10', '')
        # Style tags should be removed
        self.assertNotIn('<style', content.lower())
    
    def test_xss_11_iframe_injection(self):
        """Test blocking of iframe tag"""
        content = self.sanitized_comments.get('xss_11', '')
        self.assertNotIn('<iframe', content.lower())
    
    def test_xss_12_object_embed(self):
        """Test blocking of object/embed tags"""
        content = self.sanitized_comments.get('xss_12', '')
        self.assertNotIn('<object', content.lower())
        self.assertNotIn('<embed', content.lower())
    
    def test_xss_13_meta_refresh(self):
        """Test blocking of meta refresh redirect"""
        content = self.sanitized_comments.get('xss_13', '')
        self.assertNotIn('<meta', content.lower())
    
    def test_xss_14_entity_encoded(self):
        """Test blocking of entity-encoded script"""
        content = self.sanitized_comments.get('xss_14', '')
        self.assertNoScriptTags(content)
        # Check that even if decoded, no script exists
        self.assertNotIn('alert', content.lower())
    
    def test_xss_15_nested_script(self):
        """Test blocking of nested/obfuscated script tags"""
        content = self.sanitized_comments.get('xss_15', '')
        self.assertNoScriptTags(content)
    
    # Legitimate Formatting Tests (5 tests)
    
    def test_legitimate_01_basic_formatting(self):
        """Test preservation of basic text formatting (bold, italic)"""
        content = self.sanitized_comments.get('legitimate_01', '')
        self.assertIn('<b>', content.lower() or '<strong>' in content.lower())
        self.assertIn('<i>', content.lower() or '<em>' in content.lower())
        # Check text content is preserved
        self.assertTrue(len(content) > 0, "Content should not be empty")
    
    def test_legitimate_02_paragraph_breaks(self):
        """Test preservation of paragraphs and line breaks"""
        content = self.sanitized_comments.get('legitimate_02', '')
        # Should have paragraph or br tags
        has_structure = '<p>' in content.lower() or '<br' in content.lower()
        self.assertTrue(has_structure, "Paragraph structure should be preserved")
    
    def test_legitimate_03_safe_link(self):
        """Test preservation of legitimate HTTPS links"""
        content = self.sanitized_comments.get('legitimate_03', '')
        self.assertIn('<a', content.lower())
        self.assertIn('href', content.lower())
        # Should contain https URL
        self.assertTrue('https://' in content or 'http://' in content)
        # Should not have javascript
        self.assertNoJavaScriptURL(content)
    
    def test_legitimate_04_lists(self):
        """Test preservation of list structures"""
        content = self.sanitized_comments.get('legitimate_04', '')
        # Should have list tags
        has_list = '<ul>' in content.lower() or '<ol>' in content.lower()
        self.assertTrue(has_list, "List structure should be preserved")
        self.assertIn('<li>', content.lower())
    
    def test_legitimate_05_mixed_formatting(self):
        """Test preservation of mixed safe HTML elements"""
        content = self.sanitized_comments.get('legitimate_05', '')
        # Should preserve multiple formatting elements
        self.assertTrue(len(content) > 0, "Content should not be empty")
        # Check that content has some HTML structure
        self.assertIn('<', content)
        self.assertIn('>', content)
        # Should not have any dangerous content
        self.assertNoScriptTags(content)
        self.assertNoEventHandlers(content)
        self.assertNoJavaScriptURL(content)

if __name__ == '__main__':
    unittest.main()