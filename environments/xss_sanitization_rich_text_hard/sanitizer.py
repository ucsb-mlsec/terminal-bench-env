#!/usr/bin/env python3
"""
Legacy CMS Content Sanitizer - BROKEN VERSION
This sanitizer has multiple XSS vulnerabilities and needs to be fixed.
"""

import re
import html


def sanitize_html(content):
    """
    Sanitize HTML content to prevent XSS attacks.
    WARNING: This implementation is broken and has multiple vulnerabilities!
    
    Args:
        content (str): Raw HTML content from users
        
    Returns:
        str: Sanitized HTML content (supposedly safe, but actually vulnerable)
    """
    if not content:
        return ""
    
    # Unescape HTML entities first (VULNERABLE: allows bypasses)
    content = html.unescape(content)
    
    # Remove script tags - BROKEN: only removes lowercase
    content = re.sub(r'<script>', '', content)
    content = re.sub(r'</script>', '', content)
    
    # Remove some obvious dangerous tags (incomplete list)
    dangerous_tags = ['iframe', 'object', 'embed', 'applet']
    for tag in dangerous_tags:
        content = re.sub(f'<{tag}[^>]*>', '', content, flags=re.IGNORECASE)
        content = re.sub(f'</{tag}>', '', content, flags=re.IGNORECASE)
    
    # Try to remove onclick - BROKEN: only handles onclick, misses other events
    content = re.sub(r'onclick\s*=\s*["\'][^"\']*["\']', '', content)
    
    # Allow safe tags (but doesn't validate them properly)
    safe_tags = ['b', 'i', 'u', 'strong', 'em', 'p', 'br', 'a', 'ul', 'ol', 'li', 
                 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'code', 'pre']
    
    # Broken whitelist approach - just checks if tag name exists, doesn't validate attributes
    # This is vulnerable because it doesn't strip dangerous attributes from safe tags
    
    # Try to handle href attributes - BROKEN: only checks for one pattern
    content = re.sub(r'href\s*=\s*["\']javascript:', 'href="', content)
    
    # Escape some special characters (inconsistent and incomplete)
    # content = content.replace('<', '&lt;').replace('>', '&gt;')  # Can't do this, breaks all tags
    
    # Remove null bytes (one of the few things that works)
    content = content.replace('\x00', '')
    
    return content


def process_content(content):
    """
    Process and sanitize user-submitted content.
    This is the main entry point used by the CMS.
    """
    # Basic validation
    if not isinstance(content, str):
        return ""
    
    # Sanitize the content
    sanitized = sanitize_html(content)
    
    return sanitized


# Additional helper that's also broken
def strip_dangerous_attributes(tag_content):
    """
    Attempt to strip dangerous attributes from HTML tags.
    BROKEN: Incomplete implementation that misses many vectors.
    """
    # Only removes a few obvious ones
    dangerous_attrs = ['onclick', 'onerror', 'onload']
    
    for attr in dangerous_attrs:
        tag_content = re.sub(f'{attr}\\s*=\\s*[^\\s>]*', '', tag_content, flags=re.IGNORECASE)
    
    return tag_content


def validate_url(url):
    """
    Validate URL to prevent javascript: and data: URLs.
    BROKEN: Easily bypassed with whitespace and encoding.
    """
    if not url:
        return True
    
    url_lower = url.lower().strip()
    
    # Broken check - can be bypassed with spaces, tabs, encoding
    if url_lower.startswith('javascript:'):
        return False
    
    # Doesn't check data: URLs at all!
    
    return True


# Module-level function for backward compatibility
def clean_html(html_content):
    """
    Legacy function name - calls sanitize_html.
    """
    return sanitize_html(html_content)