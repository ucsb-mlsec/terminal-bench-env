#!/usr/bin/env python3
"""
textproc - A simple text processing command-line utility
"""

import click
import requests
import sys
from pathlib import Path


def count_text_stats(content):
    """
    Count basic statistics for text content.
    
    Args:
        content: String content to analyze
        
    Returns:
        Dictionary with text statistics
    """
    lines = content.split('\n')
    words = content.split()
    chars = len(content)
    
    return {
        'lines': len(lines),
        'words': len(words),
        'characters': chars
    }


def check_api_status(url):
    """
    Check if a URL is accessible.
    
    Args:
        url: URL to check
        
    Returns:
        Status code or None if failed
    """
    try:
        response = requests.get(url, timeout=5)
        return response.status_code
    except requests.RequestException:
        return None


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--check-url', '-u', help='Optional URL to check status')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(filename, check_url, verbose):
    """
    Process text file and display statistics.
    
    FILENAME: Path to the text file to process
    """
    try:
        file_path = Path(filename)
        content = file_path.read_text()
        
        stats = count_text_stats(content)
        
        click.echo(f"File: {filename}")
        click.echo(f"Lines: {stats['lines']}")
        click.echo(f"Words: {stats['words']}")
        click.echo(f"Characters: {stats['characters']}")
        
        if check_url:
            status = check_api_status(check_url)
            if status:
                click.echo(f"URL Status: {status}")
            else:
                click.echo("URL check failed")
        
        if verbose:
            click.echo("\nFirst 100 characters:")
            click.echo(content[:100])
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()