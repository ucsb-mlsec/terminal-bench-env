#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

class ChurnPredictionPipeline:
    """Customer churn prediction pipeline with preprocessing and classification."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.feature_columns = ['age', 'tenure_months', 'monthly_spend', 'support_calls']
    
    def fit(self, X, y):
        """Train the pipeline on customer data."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        """Make churn predictions."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Return prediction probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        return self.pipeline.predict_proba(X)


def train_pipeline(data_path):
    """Load data and train the churn prediction pipeline."""
    # Load customer data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df[['age', 'tenure_months', 'monthly_spend', 'support_calls']]
    y = df['churned']
    
    # Create and train pipeline
    model = ChurnPredictionPipeline()
    model.fit(X, y)
    
    return model
```

```python
#!/usr/bin/env python3

import pandas as pd
import pickle
import sys
sys.path.insert(0, '/workspace')

from train_model import ChurnPredictionPipeline, train_pipeline

def main():
    # Train the pipeline using the training script
    print("Training pipeline...")
    model = train_pipeline('/workspace/data/customers.csv')
    
    # Save the trained pipeline
    print("Saving pipeline to /workspace/model.pkl...")
    with open('/workspace/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Load the saved pipeline to verify it works
    print("Loading pipeline to verify...")
    with open('/workspace/model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Make a test prediction
    print("Making test prediction...")
    test_data = pd.DataFrame({
        'age': [35],
        'tenure_months': [24],
        'monthly_spend': [89.50],
        'support_calls': [3]
    })
    
    prediction = loaded_model.predict(test_data)[0]
    
    # Write the prediction to result.txt
    print(f"Writing prediction result: {prediction}")
    with open('/workspace/result.txt', 'w') as f:
        f.write(str(prediction) + '\n')
    
    print("Pipeline rebuild complete!")

if __name__ == '__main__':
    main()