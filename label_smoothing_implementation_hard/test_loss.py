#!/usr/bin/env python3

import unittest
import numpy as np
from loss import cross_entropy_loss


class TestLossFunction(unittest.TestCase):
    
    def setUp(self):
        """Set up common test data."""
        np.random.seed(42)
        self.num_classes = 10
        self.epsilon = 1e-7
    
    def test_single_sample(self):
        """Test loss computation with batch_size=1."""
        predictions = np.random.rand(1, self.num_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        targets = np.array([3])
        
        loss = cross_entropy_loss(predictions, targets)
        
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
    
    def test_batch_processing(self):
        """Test with batch_size=32."""
        batch_size = 32
        predictions = np.random.rand(batch_size, self.num_classes)
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        targets = np.random.randint(0, self.num_classes, size=batch_size)
        
        loss = cross_entropy_loss(predictions, targets)
        
        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
    
    def test_perfect_predictions(self):
        """Test when predictions exactly match targets."""
        batch_size = 16
        targets = np.random.randint(0, self.num_classes, size=batch_size)
        predictions = np.zeros((batch_size, self.num_classes))
        for i, target in enumerate(targets):
            predictions[i, target] = 1.0
        
        loss = cross_entropy_loss(predictions, targets)
        
        # With perfect predictions, loss should be very small (near zero)
        # Allow for some tolerance due to smoothing
        self.assertLess(loss, 0.5)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
    
    def test_worst_predictions(self):
        """Test when predictions are opposite of targets."""
        batch_size = 16
        targets = np.array([0] * batch_size)
        predictions = np.zeros((batch_size, self.num_classes))
        # Assign high probability to wrong classes
        for i in range(batch_size):
            predictions[i, 1:] = 1.0 / (self.num_classes - 1)
        
        loss_worst = cross_entropy_loss(predictions, targets)
        
        # Create better predictions for comparison
        predictions_better = np.zeros((batch_size, self.num_classes))
        for i in range(batch_size):
            predictions_better[i, 0] = 0.5
            predictions_better[i, 1:] = 0.5 / (self.num_classes - 1)
        
        loss_better = cross_entropy_loss(predictions_better, targets)
        
        # Worst predictions should give higher loss
        self.assertGreater(loss_worst, loss_better)
        self.assertFalse(np.isnan(loss_worst))
        self.assertFalse(np.isinf(loss_worst))
    
    def test_numerical_stability(self):
        """Test with very small prediction values."""
        batch_size = 8
        targets = np.random.randint(0, self.num_classes, size=batch_size)
        predictions = np.full((batch_size, self.num_classes), self.epsilon)
        for i, target in enumerate(targets):
            predictions[i, target] = 1.0 - (self.num_classes - 1) * self.epsilon
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
        
        loss = cross_entropy_loss(predictions, targets)
        
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))
        self.assertGreater(loss, 0)
    
    def test_different_classes(self):
        """Test with different target classes to ensure all classes work."""
        test_classes = [0, 5, 9]
        losses = []
        
        for target_class in test_classes:
            predictions = np.random.rand(5, self.num_classes)
            predictions = predictions / predictions.sum(axis=1, keepdims=True)
            targets = np.array([target_class] * 5)
            
            loss = cross_entropy_loss(predictions, targets)
            
            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isinf(loss))
            self.assertGreater(loss, 0)
            losses.append(loss)
        
        # All losses should be valid positive numbers
        self.assertEqual(len(losses), len(test_classes))
        for loss in losses:
            self.assertGreater(loss, 0)


if __name__ == '__main__':
    unittest.main()