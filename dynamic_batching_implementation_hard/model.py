import numpy as np
import time

class Predictor:
    def __init__(self):
        """Initialize the predictor model."""
        # Simulate model loading time
        time.sleep(0.5)
        self.model_loaded = True
    
    def predict(self, input_data):
        """
        Run prediction on input tensor(s).
        
        Args:
            input_data: Either a single numpy array or a list of numpy arrays
            
        Returns:
            Single float value for single input, or list of floats for batch input
        """
        # Check if input is a list (batch processing)
        if isinstance(input_data, list):
            # Batch processing - much faster per tensor
            results = []
            for tensor in input_data:
                # Simulate faster batch processing time
                time.sleep(0.01)
                # Generate deterministic prediction based on tensor data
                prediction = self._compute_prediction(tensor)
                results.append(prediction)
            return results
        else:
            # Single tensor processing - slower
            time.sleep(0.08)
            # Generate deterministic prediction
            prediction = self._compute_prediction(input_data)
            return prediction
    
    def _compute_prediction(self, tensor):
        """
        Compute a deterministic prediction value from a tensor.
        
        Args:
            tensor: numpy array
            
        Returns:
            float: prediction value
        """
        # Use a deterministic formula based on tensor statistics
        # This ensures the same tensor always produces the same result
        mean_val = np.mean(tensor)
        std_val = np.std(tensor)
        sum_val = np.sum(tensor)
        
        # Combine statistics in a deterministic way
        prediction = (mean_val * 10.0 + std_val * 5.0 + sum_val * 0.001) % 100.0
        
        return float(prediction)