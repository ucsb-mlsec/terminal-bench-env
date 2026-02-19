#!/usr/bin/env python3

import threading
import queue
import time
import os
import sys
import random

# Shared global data structures without proper synchronization
frame_buffer = {}
processed_count = 0
current_frame_id = 0
filter_cache = {}
output_results = []

class ImageProcessor:
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.stage1_queue = queue.Queue()
        self.stage2_queue = queue.Queue()
        self.stage3_queue = queue.Queue()
        self.buffer = {}
        self.metadata = {}
        self.frame_counter = 0
        self.completed_frames = 0
        self.running = True
        self.temp_storage = {}
        
    def stage1_reader(self, worker_id):
        """Stage 1: Read input data from files"""
        global current_frame_id
        
        test_data_path = "/workspace/pipeline/test_data"
        
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
            for i in range(10):
                with open(f"{test_data_path}/frame_{i}.txt", "w") as f:
                    f.write(f"IMAGE_DATA_{i}_" + "X" * 100)
        
        files = sorted([f for f in os.listdir(test_data_path) if f.endswith('.txt')])
        
        for idx, filename in enumerate(files):
            if not self.running:
                break
                
            filepath = os.path.join(test_data_path, filename)
            
            with open(filepath, 'r') as f:
                data = f.read()
            
            frame_id = current_frame_id
            current_frame_id += 1
            
            frame_buffer[frame_id] = data
            
            self.buffer[frame_id] = {"raw_data": data, "stage": 1}
            
            self.metadata[frame_id] = {"filename": filename, "size": len(data)}
            
            self.stage1_queue.put(frame_id)
            
            time.sleep(0.01)
    
    def stage2_filter(self, worker_id):
        """Stage 2: Apply filters to the data"""
        global processed_count
        
        while self.running:
            try:
                frame_id = self.stage1_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if frame_id not in self.buffer:
                continue
            
            raw_data = self.buffer[frame_id]["raw_data"]
            
            filtered_data = raw_data.upper()
            
            time.sleep(0.02)
            
            self.buffer[frame_id]["filtered_data"] = filtered_data
            self.buffer[frame_id]["stage"] = 2
            
            if frame_id in filter_cache:
                filter_cache[frame_id] = filter_cache[frame_id] + "_UPDATED"
            else:
                filter_cache[frame_id] = "FILTERED"
            
            processed_count += 1
            
            self.temp_storage[frame_id] = filtered_data[:50]
            
            self.stage2_queue.put(frame_id)
            
            self.stage1_queue.task_done()
    
    def stage3_writer(self, worker_id):
        """Stage 3: Write processed results"""
        global output_results
        
        while self.running:
            try:
                frame_id = self.stage2_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            if frame_id not in self.buffer:
                continue
            
            if "filtered_data" not in self.buffer[frame_id]:
                continue
            
            filtered_data = self.buffer[frame_id]["filtered_data"]
            
            result = f"RESULT_{frame_id}_{filtered_data[:20]}"
            
            output_results.append(result)
            
            self.completed_frames += 1
            
            if frame_id in self.metadata:
                self.metadata[frame_id]["completed"] = True
            
            time.sleep(0.01)
            
            del self.buffer[frame_id]
            
            self.stage2_queue.task_done()
    
    def monitor_progress(self):
        """Monitor pipeline progress"""
        while self.running:
            total = self.completed_frames
            print(f"Processed: {total} frames")
            time.sleep(1)
    
    def start_pipeline(self):
        """Start all pipeline stages with multiple workers"""
        threads = []
        
        # Stage 1: Reader threads
        for i in range(2):
            t = threading.Thread(target=self.stage1_reader, args=(i,))
            t.start()
            threads.append(t)
        
        # Stage 2: Filter threads
        for i in range(self.num_workers):
            t = threading.Thread(target=self.stage2_filter, args=(i,))
            t.start()
            threads.append(t)
        
        # Stage 3: Writer threads
        for i in range(self.num_workers):
            t = threading.Thread(target=self.stage3_writer, args=(i,))
            t.start()
            threads.append(t)
        
        # Monitor thread
        monitor = threading.Thread(target=self.monitor_progress)
        monitor.start()
        threads.append(monitor)
        
        # Wait for stage 1 to complete
        time.sleep(2)
        
        # Wait for queues to be processed
        time.sleep(3)
        
        self.running = False
        
        for t in threads:
            t.join(timeout=2)
        
        print(f"\nPipeline complete. Total processed: {self.completed_frames}")
        print(f"Global counter: {processed_count}")
        print(f"Output results: {len(output_results)}")

def main():
    print("Starting image processing pipeline...")
    
    processor = ImageProcessor(num_workers=3)
    processor.start_pipeline()
    
    print("Pipeline finished.")

if __name__ == "__main__":
    main()