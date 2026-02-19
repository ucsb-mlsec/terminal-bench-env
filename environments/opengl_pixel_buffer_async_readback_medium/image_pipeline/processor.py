#!/usr/bin/env python3

import numpy as np
import time
import os
import sys

class ImageProcessor:
    """
    Synchronous image processing pipeline.
    Currently suffers from severe performance degradation due to blocking I/O.
    """
    
    def __init__(self, output_dir='/workspace/output'):
        self.output_dir = output_dir
        self.frame_width = 512
        self.frame_height = 512
        self.channels = 3
        
    def setup(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_frame(self, frame_number):
        """
        Generate synthetic image data as numpy array.
        Returns a 512x512x3 RGB image with random data.
        """
        # Generate random RGB image data
        frame = np.random.randint(0, 256, 
                                 (self.frame_height, self.frame_width, self.channels),
                                 dtype=np.uint8)
        return frame
    
    def apply_transformation(self, frame):
        """
        Apply CPU-bound transformations to the frame.
        Simulates processing operations like brightness/contrast adjustment.
        Takes approximately 20-50ms depending on system.
        """
        # Simulate CPU-bound processing
        processed = frame.astype(np.float32)
        
        # Brightness adjustment
        processed = processed * 1.1
        
        # Contrast enhancement
        mean = np.mean(processed)
        processed = (processed - mean) * 1.2 + mean
        
        # Simple blur filter (3x3 averaging)
        result = np.zeros_like(processed)
        for i in range(1, self.frame_height - 1):
            for j in range(1, self.frame_width - 1):
                result[i, j] = np.mean(processed[i-1:i+2, j-1:j+2], axis=(0, 1))
        
        # Clip values and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def save_frame_sync(self, frame, frame_number):
        """
        BOTTLENECK: Synchronous I/O blocks pipeline.
        
        This function performs blocking file I/O, causing the entire pipeline
        to stall while waiting for the write to complete. The sleep simulates
        slow disk writes (150-180ms), which combined with the synchronous nature
        creates severe performance degradation.
        """
        filename = os.path.join(self.output_dir, f'frame_{frame_number:04d}.npy')
        
        # Simulate slow disk I/O - this is the critical bottleneck
        # In real scenarios, this could be network storage, slow disks, etc.
        time.sleep(0.15 + np.random.uniform(0, 0.03))  # 150-180ms disk latency
        
        # Perform the actual blocking write operation
        np.save(filename, frame)
        
    def process_frame(self, frame_number):
        """
        Process a single frame through the complete pipeline.
        This is currently entirely synchronous and blocks on I/O.
        """
        # Step 1: Generate frame data
        frame = self.generate_frame(frame_number)
        
        # Step 2: Apply CPU-bound transformations
        processed_frame = self.apply_transformation(frame)
        
        # Step 3: BLOCKING write to disk - pipeline stalls here
        self.save_frame_sync(processed_frame, frame_number)
        
        return True
    
    def run_benchmark(self, num_frames=100):
        """
        Benchmark the pipeline performance.
        Processes the specified number of frames and measures throughput.
        """
        print(f"Starting synchronous benchmark with {num_frames} frames...")
        print(f"Output directory: {self.output_dir}")
        
        self.setup()
        
        start_time = time.time()
        
        # Process all frames synchronously
        for frame_num in range(num_frames):
            self.process_frame(frame_num)
            
            # Progress indicator
            if (frame_num + 1) % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = (frame_num + 1) / elapsed
                print(f"Processed {frame_num + 1}/{num_frames} frames "
                      f"({current_fps:.1f} FPS)")
        
        total_time = time.time() - start_time
        throughput = num_frames / total_time
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results (Synchronous Pipeline):")
        print(f"{'='*60}")
        print(f"Frames processed: {num_frames}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Throughput: {throughput:.1f} FPS")
        print(f"Async enabled: no")
        print(f"{'='*60}")
        
        # Save results to file
        self.save_results(num_frames, total_time, throughput, async_enabled=False)
        
        return num_frames, total_time, throughput
    
    def save_results(self, frames_processed, total_time, throughput, async_enabled):
        """Save benchmark results to results.txt file."""
        results_path = '/workspace/results.txt'
        
        with open(results_path, 'w') as f:
            f.write(f"FRAMES_PROCESSED={frames_processed}\n")
            f.write(f"TOTAL_TIME={total_time:.1f}\n")
            f.write(f"THROUGHPUT={throughput:.1f}\n")
            f.write(f"ASYNC_ENABLED={'yes' if async_enabled else 'no'}\n")
        
        print(f"\nResults saved to {results_path}")


def main():
    """Main execution entry point."""
    processor = ImageProcessor()
    processor.run_benchmark(num_frames=100)


if __name__ == '__main__':
    main()