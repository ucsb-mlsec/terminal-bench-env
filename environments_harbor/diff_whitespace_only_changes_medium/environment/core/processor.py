#!/usr/bin/env python3

def validate_record(record):
    """Validate a single record."""
    if not isinstance(record, dict):
        return False
    if 'id' not in record or 'value' not in record:
        return False
    return True


class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor."""
        self.processed_count = 0
        self.failed_count = 0
        self.cache = {}


    def process_records(self, records):
        """Process a list of records."""
        processed = []
        for record in records:
            if validate_record(record):
                processed_record = {
                    'id': record['id'],
                    'value': record['value'] * 2,
                    'processed': True
                }
                processed.append(processed_record)
                self.processed_count += 1
            else:
                self.failed_count += 1
        return processed


    def filter_data(self, data, criteria):
        """Filter data based on given criteria."""
        filtered = []
        for item in data:
            match = True
            for key, value in criteria.items():
                if key not in item or item[key] != value:
                    match = False
                    break
            if match:
                filtered.append(item)
        return filtered


    def aggregate_results(self, results):
        """Aggregate results into summary statistics."""
        if not results:
            return {'count': 0, 'total': 0, 'average': 0}
        
        total = sum(r.get('value', 0) for r in results)
        count = len(results)
        average = total / count if count > 0 else 0
        
        return {
            'count': count,
            'total': total,
            'average': average
        }
