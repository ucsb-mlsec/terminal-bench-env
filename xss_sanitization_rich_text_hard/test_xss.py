#!/usr/bin/env python3

import unittest
import time
import os
import sys

# Add the cms directory to the path
sys.path.insert(0, '/opt/cms')

from sanitizer_fixed import sanitize_html


class TestXSSSanitization(unittest.TestCase):
    """Test suite for XSS sanitization in the CMS rich text editor"""

    def test_basic_script_tag(self):
        """Test that basic script tags are removed"""
        malicious = '<p>Hello</p><script>alert("XSS")</script><p>World</p>'
        result = sanitize_html(malicious)
        self.assertNotIn('<script', result.lower())
        self.assertNotIn('alert', result)
        self.assertIn('Hello', result)
        self.assertIn('World', result)

    def test_script_tag_case_variations(self):
        """Test script tags with various case combinations"""
        test_cases = [
            '<ScRiPt>alert(1)</ScRiPt>',
            '<SCRIPT>alert(1)</SCRIPT>',
            '<sCrIpT>alert(1)</sCrIpT>',
            '<script>alert(1)</SCRIPT>',
        ]
        for malicious in test_cases:
            result = sanitize_html(malicious)
            self.assertNotIn('script', result.lower())
            self.assertNotIn('alert', result)

    def test_onclick_event_handler(self):
        """Test that onclick event handlers are removed"""
        malicious = '<button onclick="alert(1)">Click me</button>'
        result = sanitize_html(malicious)
        self.assertNotIn('onclick', result.lower())
        self.assertNotIn('alert', result)

    def test_onerror_event_handler(self):
        """Test that onerror event handlers are removed"""
        malicious = '<img src="invalid.jpg" onerror="alert(1)">'
        result = sanitize_html(malicious)
        self.assertNotIn('onerror', result.lower())
        self.assertNotIn('alert', result)

    def test_onload_event_handler(self):
        """Test that onload event handlers are removed"""
        malicious = '<body onload="alert(1)">Content</body>'
        result = sanitize_html(malicious)
        self.assertNotIn('onload', result.lower())
        self.assertNotIn('alert', result)

    def test_onmouseover_event_handler(self):
        """Test that onmouseover event handlers are removed"""
        malicious = '<div onmouseover="alert(1)">Hover me</div>'
        result = sanitize_html(malicious)
        self.assertNotIn('onmouseover', result.lower())
        self.assertNotIn('alert', result)

    def test_javascript_url_in_href(self):
        """Test that javascript: URLs in links are removed"""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        result = sanitize_html(malicious)
        self.assertNotIn('javascript:', result.lower())
        self.assertNotIn('alert', result)

    def test_javascript_url_in_img_src(self):
        """Test that javascript: URLs in img src are removed"""
        malicious = '<img src="javascript:alert(1)">'
        result = sanitize_html(malicious)
        self.assertNotIn('javascript:', result.lower())
        self.assertNotIn('alert', result)

    def test_data_url_with_script(self):
        """Test that data: URLs with scripts are blocked"""
        malicious = '<a href="data:text/html,<script>alert(1)</script>">Click</a>'
        result = sanitize_html(malicious)
        self.assertNotIn('data:', result.lower())
        self.assertNotIn('alert', result)

    def test_data_url_base64_script(self):
        """Test that base64 encoded data URLs with scripts are blocked"""
        malicious = '<iframe src="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg=="></iframe>'
        result = sanitize_html(malicious)
        self.assertNotIn('data:', result.lower())
        self.assertNotIn('iframe', result.lower())

    def test_svg_onload_xss(self):
        """Test that SVG-based XSS attacks are blocked"""
        malicious = '<svg onload="alert(1)"></svg>'
        result = sanitize_html(malicious)
        self.assertNotIn('onload', result.lower())
        self.assertNotIn('alert', result)
        self.assertNotIn('<svg', result.lower())

    def test_svg_with_script(self):
        """Test SVG with embedded script tags"""
        malicious = '<svg><script>alert(1)</script></svg>'
        result = sanitize_html(malicious)
        self.assertNotIn('<script', result.lower())
        self.assertNotIn('alert', result)

    def test_obfuscated_script_with_null_bytes(self):
        """Test obfuscated script tags with null bytes"""
        malicious = '<scr\x00ipt>alert(1)</scr\x00ipt>'
        result = sanitize_html(malicious)
        self.assertNotIn('alert', result)

    def test_obfuscated_script_with_comments(self):
        """Test script tags with HTML comments inside"""
        malicious = '<scr<!--comment-->ipt>alert(1)</scr<!---->ipt>'
        result = sanitize_html(malicious)
        self.assertNotIn('alert', result)

    def test_script_with_extra_spaces(self):
        """Test script tags with extra spaces and newlines"""
        malicious = '<  script  >alert(1)</  script  >'
        result = sanitize_html(malicious)
        self.assertNotIn('alert', result)

    def test_html_entity_encoded_script(self):
        """Test HTML entity encoding bypass attempts"""
        malicious = '&#60;script&#62;alert(1)&#60;/script&#62;'
        result = sanitize_html(malicious)
        # After decoding, script should still be blocked
        self.assertNotIn('alert', result)

    def test_hex_entity_encoded_script(self):
        """Test hex entity encoding bypass attempts"""
        malicious = '&#x3C;script&#x3E;alert(1)&#x3C;/script&#x3E;'
        result = sanitize_html(malicious)
        self.assertNotIn('alert', result)

    def test_css_expression_attack(self):
        """Test CSS expression attacks (IE-specific but should be blocked)"""
        malicious = '<div style="width: expression(alert(1))">Content</div>'
        result = sanitize_html(malicious)
        self.assertNotIn('expression', result.lower())
        self.assertNotIn('alert', result)

    def test_nested_html_exploit(self):
        """Test nested HTML that could bypass sanitization"""
        malicious = '<div><script><div></div>alert(1)</script></div>'
        result = sanitize_html(malicious)
        self.assertNotIn('<script', result.lower())
        self.assertNotIn('alert', result)

    def test_iframe_injection(self):
        """Test that iframe tags are removed"""
        malicious = '<iframe src="http://evil.com"></iframe>'
        result = sanitize_html(malicious)
        self.assertNotIn('<iframe', result.lower())

    def test_object_tag_injection(self):
        """Test that object tags are removed"""
        malicious = '<object data="http://evil.com/evil.swf"></object>'
        result = sanitize_html(malicious)
        self.assertNotIn('<object', result.lower())

    def test_embed_tag_injection(self):
        """Test that embed tags are removed"""
        malicious = '<embed src="http://evil.com/evil.swf">'
        result = sanitize_html(malicious)
        self.assertNotIn('<embed', result.lower())

    def test_legitimate_bold_formatting(self):
        """Test that legitimate bold formatting is preserved"""
        legitimate = '<p>This is <b>bold</b> text</p>'
        result = sanitize_html(legitimate)
        self.assertIn('<b>bold</b>', result)
        self.assertIn('This is', result)

    def test_legitimate_italic_formatting(self):
        """Test that legitimate italic formatting is preserved"""
        legitimate = '<p>This is <i>italic</i> text</p>'
        result = sanitize_html(legitimate)
        self.assertIn('<i>italic</i>', result)

    def test_legitimate_links(self):
        """Test that legitimate HTTP links are preserved"""
        legitimate = '<a href="https://example.com">Link</a>'
        result = sanitize_html(legitimate)
        self.assertIn('href="https://example.com"', result)
        self.assertIn('Link', result)

    def test_legitimate_lists(self):
        """Test that legitimate lists are preserved"""
        legitimate = '<ul><li>Item 1</li><li>Item 2</li></ul>'
        result = sanitize_html(legitimate)
        self.assertIn('<ul>', result)
        self.assertIn('<li>Item 1</li>', result)

    def test_legitimate_headings(self):
        """Test that legitimate headings are preserved"""
        legitimate = '<h1>Heading 1</h1><h2>Heading 2</h2>'
        result = sanitize_html(legitimate)
        self.assertIn('<h1>Heading 1</h1>', result)
        self.assertIn('<h2>Heading 2</h2>', result)

    def test_legitimate_blockquotes(self):
        """Test that legitimate blockquotes are preserved"""
        legitimate = '<blockquote>This is a quote</blockquote>'
        result = sanitize_html(legitimate)
        self.assertIn('<blockquote>', result)
        self.assertIn('This is a quote', result)

    def test_legitimate_code_blocks(self):
        """Test that legitimate code blocks are preserved"""
        legitimate = '<pre><code>print("hello")</code></pre>'
        result = sanitize_html(legitimate)
        self.assertIn('<code>', result)
        self.assertIn('print("hello")', result)

    def test_performance_1000_documents(self):
        """Test that 1000 documents can be processed in under 5 seconds"""
        sample_content = '<p>This is <b>sample</b> content with <a href="https://example.com">a link</a></p>'
        
        start_time = time.time()
        for _ in range(1000):
            result = sanitize_html(sample_content)
        end_time = time.time()
        
        elapsed = end_time - start_time
        self.assertLess(elapsed, 5.0, f"Processing took {elapsed:.2f}s, should be under 5s")

    def test_multiple_event_handlers(self):
        """Test multiple event handlers in a single tag"""
        malicious = '<div onclick="alert(1)" onmouseover="alert(2)" onload="alert(3)">Test</div>'
        result = sanitize_html(malicious)
        self.assertNotIn('onclick', result.lower())
        self.assertNotIn('onmouseover', result.lower())
        self.assertNotIn('onload', result.lower())
        self.assertNotIn('alert', result)

    def test_mixed_malicious_and_legitimate(self):
        """Test content with both malicious and legitimate elements"""
        mixed = '''
        <h1>Article Title</h1>
        <p>This is a <b>legitimate</b> paragraph.</p>
        <script>alert('XSS')</script>
        <p>Another paragraph with <a href="javascript:alert(1)">bad link</a></p>
        <p>Good link: <a href="https://example.com">Example</a></p>
        <img src="valid.jpg" onerror="alert(1)">
        '''
        result = sanitize_html(mixed)
        
        # Legitimate content should be preserved
        self.assertIn('<h1>Article Title</h1>', result)
        self.assertIn('<b>legitimate</b>', result)
        self.assertIn('href="https://example.com"', result)
        
        # Malicious content should be removed
        self.assertNotIn('<script', result.lower())
        self.assertNotIn('javascript:', result.lower())
        self.assertNotIn('onerror', result.lower())
        self.assertNotIn('alert', result)

    def test_vbscript_url(self):
        """Test that vbscript: URLs are blocked"""
        malicious = '<a href="vbscript:msgbox(1)">Click</a>'
        result = sanitize_html(malicious)
        self.assertNotIn('vbscript:', result.lower())

    def test_form_tag_injection(self):
        """Test that form tags with malicious action are handled"""
        malicious = '<form action="javascript:alert(1)"><input type="submit"></form>'
        result = sanitize_html(malicious)
        self.assertNotIn('javascript:', result.lower())

    def test_meta_refresh_redirect(self):
        """Test that meta refresh redirects are blocked"""
        malicious = '<meta http-equiv="refresh" content="0;url=javascript:alert(1)">'
        result = sanitize_html(malicious)
        self.assertNotIn('javascript:', result.lower())


if __name__ == '__main__':
    unittest.main()