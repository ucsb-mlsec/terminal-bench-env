#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MultiTaskModel, self).__init__()
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        
        # Rating prediction head (5 classes: 1-5 stars)
        self.rating_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )
        
        # Helpfulness classification head (2 classes: helpful/not_helpful)
        self.helpfulness_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        shared_features = self.shared_encoder(x)
        rating_output = self.rating_head(shared_features)
        helpfulness_output = self.helpfulness_head(shared_features)
        return rating_output, helpfulness_output

def create_sample_data():
    """Create sample reviews data if file doesn't exist"""
    reviews_data = {
        'review_text': [
            'This product is absolutely amazing! Best purchase ever.',
            'Terrible quality, broke after one use.',
            'Good value for money, works as expected.',
            'Not bad but could be better.',
            'Outstanding quality and fast delivery!',
            'Waste of money, very disappointed.',
            'Pretty decent product overall.',
            'Excellent! Highly recommend to everyone.',
            'Average product, nothing special.',
            'Very poor quality, do not buy.',
            'Great features and easy to use.',
            'Okay product but overpriced.',
            'Love it! Worth every penny.',
            'Not satisfied with the purchase.',
            'Good but has some minor issues.',
            'Perfect! Exactly what I needed.',
            'Disappointed with the quality.',
            'Nice product, will buy again.',
            'Could be better for the price.',
            'Fantastic product, very happy!'
        ],
        'rating': [5, 1, 4, 3, 5, 1, 3, 5, 3, 1, 4, 3, 5, 2, 3, 5, 2, 4, 3, 5],
        'helpfulness': ['helpful', 'helpful', 'helpful', 'not_helpful', 'helpful',
                       'helpful', 'not_helpful', 'helpful', 'not_helpful', 'helpful',
                       'helpful', 'not_helpful', 'helpful', 'helpful', 'not_helpful',
                       'helpful', 'helpful', 'helpful', 'not_helpful', 'helpful']
    }
    df = pd.DataFrame(reviews_data)
    df.to_csv('/workspace/reviews.csv', index=False)
    return df

# Load or create data
try:
    df = pd.read_csv('/workspace/reviews.csv')
except:
    df = create_sample_data()

# Preprocess text using TF-IDF
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['review_text']).toarray()

# Encode labels
rating_encoder = LabelEncoder()
helpfulness_encoder = LabelEncoder()

# Ratings are 1-5, convert to 0-4 for classification
y_rating = df['rating'].values - 1
y_helpfulness = helpfulness_encoder.fit_transform(df['helpfulness'].values)

# Train-test split
X_train, X_test, y_rating_train, y_rating_test, y_help_train, y_help_test = train_test_split(
    X, y_rating, y_helpfulness, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_rating_train_tensor = torch.LongTensor(y_rating_train)
y_rating_test_tensor = torch.LongTensor(y_rating_test)
y_help_train_tensor = torch.LongTensor(y_help_train)
y_help_test_tensor = torch.LongTensor(y_help_test)

# Initialize model
input_dim = X_train.shape[1]
model = MultiTaskModel(input_dim=input_dim)

# Loss functions
rating_criterion = nn.CrossEntropyLoss()
helpfulness_criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CRITICAL ISSUE: Severely imbalanced loss weights
# This causes one task to dominate training
RATING_WEIGHT = 1.0
HELPFULNESS_WEIGHT = 0.001  # This is severely underweighted!

# Training parameters
num_epochs = 80
print_every = 20

# Track initial losses
model.eval()
with torch.no_grad():
    rating_out, help_out = model(X_train_tensor)
    initial_rating_loss = rating_criterion(rating_out, y_rating_train_tensor).item()
    initial_help_loss = helpfulness_criterion(help_out, y_help_train_tensor).item()

print(f"Initial Rating Loss: {initial_rating_loss:.4f}")
print(f"Initial Helpfulness Loss: {initial_help_loss:.4f}")
print(f"\nWarning: Loss weights are imbalanced!")
print(f"Rating weight: {RATING_WEIGHT}, Helpfulness weight: {HELPFULNESS_WEIGHT}")
print("-" * 60)

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Forward pass
    rating_output, helpfulness_output = model(X_train_tensor)
    
    # Compute individual losses
    rating_loss = rating_criterion(rating_output, y_rating_train_tensor)
    helpfulness_loss = helpfulness_criterion(helpfulness_output, y_help_train_tensor)
    
    # Combined loss with imbalanced weights
    combined_loss = RATING_WEIGHT * rating_loss + HELPFULNESS_WEIGHT * helpfulness_loss
    
    # Backward pass and optimization
    combined_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Rating Loss: {rating_loss.item():.4f}")
        print(f"  Helpfulness Loss: {helpfulness_loss.item():.4f}")
        print(f"  Combined Loss: {combined_loss.item():.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    rating_output, helpfulness_output = model(X_test_tensor)
    
    final_rating_loss = rating_criterion(rating_output, y_rating_test_tensor).item()
    final_helpfulness_loss = helpfulness_criterion(helpfulness_output, y_help_test_tensor).item()
    
    # Also compute training losses for convergence check
    rating_out_train, help_out_train = model(X_train_tensor)
    final_rating_loss_train = rating_criterion(rating_out_train, y_rating_train_tensor).item()
    final_help_loss_train = helpfulness_criterion(help_out_train, y_help_train_tensor).item()

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print(f"Rating Loss (test): {final_rating_loss:.4f}")
print(f"Helpfulness Loss (test): {final_helpfulness_loss:.4f}")

# Check convergence: both losses should decrease by at least 20%
rating_decreased = (initial_rating_loss - final_rating_loss_train) / initial_rating_loss >= 0.20
help_decreased = (initial_help_loss - final_help_loss_train) / initial_help_loss >= 0.20
converged = "yes" if (rating_decreased and help_decreased) else "no"

print(f"\nRating loss decreased by: {((initial_rating_loss - final_rating_loss_train) / initial_rating_loss * 100):.1f}%")
print(f"Helpfulness loss decreased by: {((initial_help_loss - final_help_loss_train) / initial_help_loss * 100):.1f}%")
print(f"Converged: {converged}")

# Save results
with open('/workspace/results.txt', 'w') as f:
    f.write(f"rating_loss={final_rating_loss:.3f}\n")
    f.write(f"helpfulness_loss={final_helpfulness_loss:.3f}\n")
    f.write(f"converged={converged}\n")

print("\nResults saved to /workspace/results.txt")