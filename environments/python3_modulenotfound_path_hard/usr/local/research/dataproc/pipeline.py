#!/usr/bin/env python3

from scicore import math_utils, data_ops

class ProcessingPipeline:
    def __init__(self, data):
        self.data = data
        self.preprocessed_data = None
    
    def preprocess(self):
        filtered_data = data_ops.filter_outliers(self.data)
        self.preprocessed_data = data_ops.validate_data(filtered_data)
        return self.preprocessed_data
    
    def compute_statistics(self):
        if self.preprocessed_data is None:
            self.preprocess()
        
        mean = math_utils.compute_mean(self.preprocessed_data)
        variance = math_utils.compute_variance(self.preprocessed_data)
        
        return {'mean': mean, 'variance': variance}
    
    def execute(self):
        self.preprocess()
        results = self.compute_statistics()
        return results