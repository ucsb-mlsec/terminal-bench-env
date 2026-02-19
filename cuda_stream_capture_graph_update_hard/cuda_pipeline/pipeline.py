#!/usr/bin/env python3

import time


class Stream:
    """Mock CUDA stream object"""
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.is_capturing = False
    
    def begin_capture(self):
        """Begin graph capture on this stream"""
        self.is_capturing = True
        print(f"Stream {self.stream_id}: Begin capture")
    
    def end_capture(self):
        """End graph capture and return graph"""
        self.is_capturing = False
        print(f"Stream {self.stream_id}: End capture")
        return GraphExec()


class GraphExec:
    """Mock CUDA graph executable object"""
    def __init__(self):
        self.captured = False
        self.kernel_params = None
    
    def update_params(self, data):
        """Update kernel parameters in the graph"""
        if not self.captured:
            raise RuntimeError("Cannot update graph that hasn't been captured")
        self.kernel_params = data
        print(f"Graph updated with data: {data[:3]}...")
    
    def launch(self, stream):
        """Launch the graph on the given stream"""
        if not self.captured:
            raise RuntimeError("Cannot launch graph that hasn't been captured")
        print(f"Graph launched with params: {self.kernel_params[:3] if self.kernel_params else None}...")


class CUDAGraphPipeline:
    """Pipeline that uses CUDA graphs for batch processing"""
    
    def __init__(self):
        """Initialize the pipeline with a CUDA stream"""
        self.stream = Stream(stream_id=1)
        self.graph = None
        self.initialized = False
        print("Pipeline initialized")
    
    def capture_graph(self):
        """Capture CUDA graph from stream operations - should be called once during initialization"""
        print("Capturing CUDA graph...")
        self.stream.begin_capture()
        # Simulate kernel operations that would be captured
        time.sleep(0.01)  # Simulate work
        self.graph = self.stream.end_capture()
        self.graph.captured = True
        print("Graph capture complete")
    
    def update_graph(self, batch_data):
        """Update the captured graph with new kernel parameters"""
        if self.graph is None:
            raise RuntimeError("Graph not captured yet")
        print(f"Updating graph for batch...")
        self.graph.update_params(batch_data)
    
    def execute_graph(self):
        """Execute the captured graph"""
        if self.graph is None:
            raise RuntimeError("Graph not captured yet")
        self.graph.launch(self.stream)
    
    def process_batch(self, batch_data):
        """
        Process a batch of data using CUDA graphs.
        
        Intended workflow:
        1. First time: capture the graph (initialization)
        2. For each batch: update parameters -> execute graph
        """
        if not self.initialized:
            print("First batch - initializing graph workflow")
            # BUG: Graph capture should happen BEFORE trying to update
            # but here we update first, then capture
            self.update_graph(batch_data)  # Line 87 - This is the bug!
            self.capture_graph()
            self.initialized = True
        else:
            # Normal operation: update then execute
            self.update_graph(batch_data)
        
        self.execute_graph()
        print(f"Batch processed successfully\n")


def main():
    """Simulate the production pipeline workflow"""
    print("=== CUDA Graph Pipeline Simulation ===\n")
    
    pipeline = CUDAGraphPipeline()
    
    # Simulate processing multiple batches of streaming data
    batches = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
    ]
    
    for i, batch in enumerate(batches):
        print(f"--- Processing Batch {i+1} ---")
        try:
            pipeline.process_batch(batch)
        except RuntimeError as e:
            print(f"ERROR: {e}\n")
    
    print("=== Simulation Complete ===")


if __name__ == "__main__":
    main()