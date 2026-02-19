#!/usr/bin/env python3

import os
import json
import time
import glob
import random
import base64
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
TOPICS = [
    '/var/pubsub/topics/user-events/',
    '/var/pubsub/topics/system-metrics/',
    '/var/pubsub/topics/error-logs/'
]
PROCESSED_DIR = '/var/pubsub/processed/'
DEAD_LETTER_DIR = '/var/pubsub/dead-letter/'
MAX_RETRIES = 3
RUNTIME_SECONDS = 60

class MessageSubscriber:
    def __init__(self):
        self.processed_count = 0
        self.failed_count = 0
        self.retry_counts = defaultdict(int)
        self.start_time = time.time()
        
        # Ensure output directories exist
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(DEAD_LETTER_DIR, exist_ok=True)
    
    def get_messages_ordered(self, topic_dir):
        """Get messages from a topic directory, ordered by filename for user-events"""
        pattern = os.path.join(topic_dir, '*.json')
        files = glob.glob(pattern)
        
        # For user-events, maintain order by filename (timestamp)
        if 'user-events' in topic_dir:
            files.sort()
        
        return files
    
    def process_message(self, filepath):
        """Process a single message file with error handling"""
        try:
            # Read the message file
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Parse JSON with error handling
            try:
                message = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Malformed JSON in {filepath}: {e}")
                return False
            
            # Validate message structure
            if not self.validate_message(message):
                print(f"Invalid message structure in {filepath}")
                return False
            
            # Simulate processing
            self.process_message_data(message)
            
            # Success - move to processed directory
            filename = os.path.basename(filepath)
            dest_path = os.path.join(PROCESSED_DIR, filename)
            
            # Handle potential duplicate filenames
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(PROCESSED_DIR, f"{name}_{counter}{ext}")
                counter += 1
            
            os.rename(filepath, dest_path)
            return True
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return False
    
    def validate_message(self, message):
        """Validate message has required fields"""
        required_fields = ['message_id', 'data', 'publish_time']
        return all(field in message for field in required_fields)
    
    def process_message_data(self, message):
        """Simulate actual message processing"""
        # Decode base64 data if present
        if 'data' in message and message['data']:
            try:
                base64.b64decode(message['data'])
            except Exception:
                pass  # Data might not be base64, that's ok
    
    def handle_failed_message(self, filepath):
        """Move failed message to dead letter queue after max retries"""
        filename = os.path.basename(filepath)
        dest_path = os.path.join(DEAD_LETTER_DIR, filename)
        
        # Handle potential duplicate filenames
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(DEAD_LETTER_DIR, f"{name}_{counter}{ext}")
            counter += 1
        
        try:
            os.rename(filepath, dest_path)
            self.failed_count += 1
            print(f"Moved {filename} to dead letter queue")
        except Exception as e:
            print(f"Error moving {filepath} to dead letter: {e}")
    
    def process_with_retry(self, filepath):
        """Process message with retry logic"""
        message_key = filepath
        
        # Try to process the message
        success = self.process_message(filepath)
        
        if success:
            self.processed_count += 1
            # Clean up retry count
            if message_key in self.retry_counts:
                del self.retry_counts[message_key]
            return True
        else:
            # Increment retry count
            self.retry_counts[message_key] += 1
            
            # Check if max retries exceeded
            if self.retry_counts[message_key] >= MAX_RETRIES:
                self.handle_failed_message(filepath)
                del self.retry_counts[message_key]
                return True  # Handled (moved to dead letter)
            else:
                print(f"Retry {self.retry_counts[message_key]}/{MAX_RETRIES} for {os.path.basename(filepath)}")
                return False  # Will retry later
    
    def run(self):
        """Main subscriber loop"""
        print("Starting message subscriber...")
        print(f"Monitoring topics: {TOPICS}")
        print(f"Runtime: {RUNTIME_SECONDS} seconds")
        
        processed_files = set()
        
        while True:
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Check if runtime exceeded
            if elapsed >= RUNTIME_SECONDS:
                print(f"\nRuntime limit reached ({RUNTIME_SECONDS}s)")
                break
            
            # Process messages from all topics
            for topic_dir in TOPICS:
                if not os.path.exists(topic_dir):
                    continue
                
                messages = self.get_messages_ordered(topic_dir)
                
                for filepath in messages:
                    # Skip if already processed
                    if filepath in processed_files:
                        continue
                    
                    # Process with retry logic
                    handled = self.process_with_retry(filepath)
                    
                    if handled:
                        processed_files.add(filepath)
            
            # Sleep briefly before next iteration
            time.sleep(0.1)
        
        # Calculate actual runtime
        actual_runtime = int(time.time() - self.start_time)
        
        # Write results
        self.write_results(actual_runtime)
        
        print(f"\nProcessing complete!")
        print(f"Processed: {self.processed_count}")
        print(f"Failed: {self.failed_count}")
        print(f"Runtime: {actual_runtime}s")
    
    def write_results(self, runtime):
        """Write results to output file"""
        results_path = '/home/agent/results.txt'
        with open(results_path, 'w') as f:
            f.write(f"processed={self.processed_count}\n")
            f.write(f"failed={self.failed_count}\n")
            f.write(f"runtime_seconds={runtime}\n")
        print(f"Results written to {results_path}")

if __name__ == '__main__':
    subscriber = MessageSubscriber()
    try:
        subscriber.run()
    except KeyboardInterrupt:
        print("\nSubscriber interrupted")
        actual_runtime = int(time.time() - subscriber.start_time)
        subscriber.write_results(actual_runtime)
    except Exception as e:
        print(f"Fatal error: {e}")
        actual_runtime = int(time.time() - subscriber.start_time)
        subscriber.write_results(actual_runtime)