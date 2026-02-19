#!/usr/bin/env python3

import random

class ReviewStreamGenerator:
    def __init__(self):
        self.review_count = 0
        self.max_reviews = 1000
        self.drift_point = 500
        
        # Vocabulary set A (first 500 reviews)
        self.vocab_a_positive = [
            'great', 'excellent', 'good', 'wonderful', 'fantastic', 
            'love', 'perfect', 'best', 'happy', 'satisfied'
        ]
        self.vocab_a_negative = [
            'poor', 'terrible', 'bad', 'awful', 'horrible',
            'hate', 'worst', 'disappointed', 'defective', 'broken'
        ]
        self.templates_a = [
            'This product is {adj} and works {adv}',
            'Very {adj} quality',
            '{adj} purchase, {sentiment}',
            'I am {sentiment} with this product',
            'The product is {adj}'
        ]
        
        # Vocabulary set B (reviews 501-1000)
        self.vocab_b_positive = [
            'amazing', 'outstanding', 'superb', 'brilliant', 'exceptional',
            'recommend', 'impressive', 'reliable', 'premium', 'top-notch'
        ]
        self.vocab_b_negative = [
            'disappointing', 'unreliable', 'mediocre', 'inferior', 'substandard',
            'frustrating', 'useless', 'waste', 'regret', 'faulty'
        ]
        self.templates_b = [
            '{adj} performance, highly {sentiment} for daily use',
            '{adj} experience, completely {adv}',
            'After weeks of use, I find it {adj}',
            '{sentiment} this product to anyone looking for quality',
            'The build quality is {adj} and {sentiment}'
        ]
        
        random.seed(42)  # For reproducibility
        
    def _generate_review_a(self, label):
        """Generate review using vocabulary set A"""
        if label == 1:  # Positive
            adj = random.choice(self.vocab_a_positive)
            template = random.choice(self.templates_a)
            
            if '{adv}' in template:
                text = template.format(adj=adj, adv='well')
            elif '{sentiment}' in template:
                text = template.format(adj=adj, sentiment='very pleased')
            else:
                text = template.format(adj=adj)
        else:  # Negative
            adj = random.choice(self.vocab_a_negative)
            template = random.choice(self.templates_a)
            
            if '{adv}' in template:
                text = template.format(adj=adj, adv='poorly')
            elif '{sentiment}' in template:
                text = template.format(adj=adj, sentiment='very unhappy')
            else:
                text = template.format(adj=adj)
                
        return text
    
    def _generate_review_b(self, label):
        """Generate review using vocabulary set B"""
        if label == 1:  # Positive
            adj = random.choice(self.vocab_b_positive)
            template = random.choice(self.templates_b)
            
            if '{adv}' in template:
                text = template.format(adj=adj, adv='dependable')
            elif '{sentiment}' in template:
                if 'recommend' in template:
                    text = template.format(sentiment='would recommend')
                else:
                    text = template.format(adj=adj, sentiment='satisfying')
            else:
                text = template.format(adj=adj)
        else:  # Negative
            adj = random.choice(self.vocab_b_negative)
            template = random.choice(self.templates_b)
            
            if '{adv}' in template:
                text = template.format(adj=adj, adv='unreliable')
            elif '{sentiment}' in template:
                if 'recommend' in template:
                    text = template.format(sentiment='cannot recommend')
                else:
                    text = template.format(adj=adj, sentiment='frustrating')
            else:
                text = template.format(adj=adj)
                
        return text
    
    def generate_review(self):
        """Generate next review in the stream"""
        if self.review_count >= self.max_reviews:
            raise StopIteration("All reviews have been generated")
        
        # Roughly balanced labels
        label = 1 if random.random() > 0.5 else 0
        
        # Choose vocabulary set based on drift point
        if self.review_count < self.drift_point:
            text = self._generate_review_a(label)
        else:
            text = self._generate_review_b(label)
        
        self.review_count += 1
        return (text, label)
    
    def reset(self):
        """Reset the generator to start from beginning"""
        self.review_count = 0
        random.seed(42)


# Global generator instance
_generator = ReviewStreamGenerator()

def generate_review():
    """
    Generate one review at a time from the stream.
    
    Returns:
        tuple: (text, label) where text is a string and label is 0 or 1
    
    Raises:
        StopIteration: When all 1000 reviews have been generated
    """
    return _generator.generate_review()

def reset_generator():
    """Reset the generator (useful for testing)"""
    global _generator
    _generator = ReviewStreamGenerator()