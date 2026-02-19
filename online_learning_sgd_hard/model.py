#!/usr/bin/env python3

import numpy as np

class OnlinePredictor:
    """
    Online SGD predictor for streaming data.
    
    TODO: Fix learning rate - might be too high and causes instability
    FIXME: Need to add gradient clipping to prevent NaN issues
    FIXME: Memory usage keeps growing - history list never gets cleared
    """
    
    def __init__(self, n_features):
        # Initialize weights - starting with zeros might not be ideal
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Fixed learning rate - no decay mechanism
        self.learning_rate = 0.1  # TODO: This seems too high, add decay?
        
        # Store all predictions and errors for debugging
        # FIXME: This will cause memory issues in production!
        self.history = []
        self.error_history = []
        self.prediction_history = []
        
        self.n_updates = 0
    
    def predict(self, features):
        """Make a prediction for given features."""
        features = np.array(features)
        prediction = np.dot(self.weights, features) + self.bias
        
        # Store prediction for later analysis
        self.prediction_history.append(prediction)
        
        return prediction
    
    def update(self, features, target):
        """
        Update model parameters using SGD.
        
        TODO: Add regularization to prevent weights from exploding
        FIXME: Gradients can get very large, need clipping
        """
        features = np.array(features)
        
        # Make prediction
        pred = self.predict(features)
        
        # Calculate error
        error = pred - target
        
        # Store error - this list grows forever!
        self.error_history.append(error)
        self.history.append({'pred': pred, 'target': target, 'error': error})
        
        # Compute gradients (no clipping applied)
        grad_weights = error * features
        grad_bias = error
        
        # Update parameters - no regularization term
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        
        self.n_updates += 1
    
    def get_stats(self):
        """Get statistics about model performance."""
        if len(self.error_history) == 0:
            return {'mean_error': 0, 'std_error': 0}
        
        errors = np.array(self.error_history)
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'n_updates': self.n_updates
        }


if __name__ == "__main__":
    # Basic test
    model = OnlinePredictor(n_features=5)
    
    # Simulate some data
    for i in range(100):
        features = np.random.randn(5)
        target = np.sum(features) * 0.5 + 1.0
        
        pred = model.predict(features)
        model.update(features, target)
        
        if i % 20 == 0:
            stats = model.get_stats()
            print(f"Update {i}: mean_error={stats['mean_error']:.4f}")
    
    print(f"\nFinal stats: {model.get_stats()}")
    print(f"History size: {len(model.history)} entries")  # This keeps growing!