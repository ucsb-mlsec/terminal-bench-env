#!/usr/bin/env python3

import sys
import os
import configparser
import re
from datetime import datetime
from pathlib import Path

class LogAnalyzer:
    def __init__(self, config_path="/etc/loganalyzer/config.ini"):
        """Initialize the log analyzer with configuration."""
        self.config = None
        self.log_dir = "/var/logs/application/"
        self.patterns = []
        self.output_file = None
        
        # Try to load configuration - poor error handling
        try:
            self.load_config(config_path)
        except:
            # Silently exit on any error - bad practice but realistic
            sys.exit(1)
    
    def load_config(self, config_path):
        """Load configuration from file."""
        # This will fail if config doesn't exist
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        parser = configparser.ConfigParser()
        parser.read_string(config_content)
        
        self.config = parser
        
        # Load patterns from config
        if 'patterns' in parser:
            for key in parser['patterns']:
                self.patterns.append(parser['patterns'][key])
        
        # Load output settings
        if 'output' in parser:
            self.output_file = parser['output'].get('file', '/var/log/analysis.log')
    
    def scan_logs(self):
        """Scan log files in the configured directory."""
        log_files = []
        
        if not os.path.exists(self.log_dir):
            print(f"Log directory {self.log_dir} not found")
            return []
        
        for root, dirs, files in os.walk(self.log_dir):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
        
        return log_files
    
    def analyze_file(self, filepath):
        """Analyze a single log file."""
        results = []
        
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern in self.patterns:
                        if re.search(pattern, line):
                            results.append({
                                'file': filepath,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
        
        return results
    
    def generate_report(self, results):
        """Generate analysis report."""
        report = []
        report.append("=" * 80)
        report.append(f"Log Analysis Report - {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append(f"\nTotal matches found: {len(results)}\n")
        
        for result in results:
            report.append(f"File: {result['file']}")
            report.append(f"Line: {result['line']}")
            report.append(f"Pattern: {result['pattern']}")
            report.append(f"Content: {result['content']}")
            report.append("-" * 80)
        
        return "\n".join(report)
    
    def run(self):
        """Main execution method."""
        print("Starting log analysis...")
        
        log_files = self.scan_logs()
        print(f"Found {len(log_files)} log files")
        
        all_results = []
        for log_file in log_files:
            results = self.analyze_file(log_file)
            all_results.extend(results)
        
        report = self.generate_report(all_results)
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(report)
            print(f"Report written to {self.output_file}")
        else:
            print(report)
        
        print("Analysis complete")

def main():
    """Main entry point."""
    analyzer = LogAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()