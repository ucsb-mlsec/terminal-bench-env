#!/usr/bin/env python3
"""
Data Formatter Module for Genomic Analysis

This module handles formatting of genomic analysis data into various output
formats including JSON, CSV, and formatted reports for the genomics pipeline.
"""

import json
import csv
from io import StringIO


def to_json(data, indent=2):
    """
    Format genomic analysis results to JSON format.
    
    Args:
        data (dict): Dictionary containing analysis results
        indent (int): Number of spaces for JSON indentation
    
    Returns:
        str: JSON formatted string of the data
    """
    return json.dumps(data, indent=indent, default=str)


def to_csv(data, headers=None):
    """
    Format genomic analysis results to CSV format.
    
    Args:
        data (list): List of dictionaries or list of lists containing results
        headers (list): Optional list of column headers
    
    Returns:
        str: CSV formatted string of the data
    """
    output = StringIO()
    
    if not data:
        return ""
    
    if isinstance(data[0], dict):
        headers = headers or list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    else:
        writer = csv.writer(output)
        if headers:
            writer.writerow(headers)
        writer.writerows(data)
    
    return output.getvalue()


def generate_report(results, title="Genomic Analysis Report"):
    """
    Generate a formatted summary report from analysis results.
    
    Args:
        results (dict): Dictionary containing analysis results and statistics
        title (str): Title for the report
    
    Returns:
        str: Formatted text report
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(title.center(60))
    report_lines.append("=" * 60)
    report_lines.append("")
    
    for key, value in results.items():
        report_lines.append(f"{key}: {value}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)


def format_summary(statistics):
    """
    Format statistical summary of genomic data.
    
    Args:
        statistics (dict): Dictionary of statistical values
    
    Returns:
        str: Formatted summary string
    """
    summary = "Summary Statistics:\n"
    summary += "-" * 40 + "\n"
    
    for stat_name, stat_value in statistics.items():
        summary += f"  {stat_name}: {stat_value}\n"
    
    return summary