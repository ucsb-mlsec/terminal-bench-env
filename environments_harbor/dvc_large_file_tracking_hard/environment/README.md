# Customer Segmentation ML Pipeline

## Overview

This project implements a comprehensive machine learning pipeline for customer segmentation and churn prediction. The system processes customer transaction data, behavioral metrics, and demographic information to identify distinct customer segments and predict churn probability. The pipeline is designed to handle large-scale datasets and includes automated preprocessing, feature engineering, and model training stages.

The solution uses ensemble methods, primarily RandomForest classifiers, to achieve robust predictions across diverse customer profiles. The modular architecture allows for easy experimentation with different feature sets and model configurations.

## Project Structure

The project is organized into the following directory structure:

### data/
Contains all data files across different processing stages:
- **raw/**: Original, unprocessed datasets from source systems
- **preprocessed/**: Cleaned and normalized data ready for feature engineering
- **features/**: Engineered feature sets in optimized formats

### models/
Stores trained model artifacts:
- Serialized model files (.pkl, .joblib)
- Model metadata and performance metrics
- Checkpoint files for iterative training

### config/
Configuration files for the pipeline:
- Data schema definitions
- Model hyperparameters
- Feature engineering specifications
- Environment-specific settings

### scripts/
Python scripts implementing the pipeline stages:
- Data preprocessing utilities
- Feature engineering transformations
- Model training and evaluation
- Prediction and inference modules

## Data Pipeline

The ML pipeline consists of three primary stages:

### Stage 1: Data Preprocessing
**Script**: `scripts/preprocess.py`

This stage handles the initial data cleaning and preparation:
- Reads raw customer data from `data/raw/raw_data.csv`
- Performs data quality checks and validation
- Handles missing values using multiple imputation strategies
- Removes duplicates and resolves data conflicts
- Normalizes numerical features
- Encodes categorical variables
- Outputs cleaned data to `data/preprocessed/preprocessed_data.csv`

Key operations include outlier detection, date parsing, and standardization of customer identifiers across different data sources.

### Stage 2: Feature Engineering
**Script**: `scripts/feature_engineering.py`

Transforms preprocessed data into ML-ready features:
- Loads preprocessed data from previous stage
- Creates temporal features (recency, frequency, monetary value)
- Generates behavioral aggregations
- Calculates customer lifetime value metrics
- Produces interaction features between key variables
- Applies dimensionality reduction where appropriate
- Saves optimized feature set to `data/features/features.parquet`

The feature set includes over 80 engineered features capturing customer behavior patterns, purchase trends, and engagement metrics.

### Stage 3: Model Training
**Script**: `scripts/train.py`

Trains and validates the segmentation models:
- Loads feature data from `data/features/`
- Splits data into training and validation sets
- Trains RandomForest classifier for customer segmentation
- Trains separate model for churn prediction
- Performs hyperparameter optimization
- Evaluates model performance using cross-validation
- Saves trained models to `models/` directory
- Generates performance reports and feature importance analysis

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git for version control
- DVC for data version control

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml_project
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize DVC (if not already configured):
```bash
dvc init
```

4. Configure DVC remote storage:
```bash
dvc remote add -d storage /path/to/dvc/storage
```

5. Pull data from DVC remote:
```bash
dvc pull
```

This will download all tracked data files and model artifacts from the configured remote storage location.

### Data Download

If you're setting up for the first time and data is not available in DVC:
- Contact the data team for access to raw datasets
- Place raw data files in `data/raw/` directory
- Run the preprocessing pipeline to generate derived datasets

## Usage

### Running the Complete Pipeline

Execute the entire pipeline from raw data to trained models:

```bash
# Step 1: Preprocess raw data
python scripts/preprocess.py --input data/raw/raw_data.csv --output data/preprocessed/preprocessed_data.csv

# Step 2: Engineer features
python scripts/feature_engineering.py --input data/preprocessed/preprocessed_data.csv --output data/features/features.parquet

# Step 3: Train models
python scripts/train.py --features data/features/features.parquet --output models/ --config config/model_config.yaml
```

### Running Individual Stages

Each script can be run independently with custom parameters:

**Preprocessing with custom configuration:**
```bash
python scripts/preprocess.py --config config/preprocess_config.yaml
```

**Feature engineering with specific feature sets:**
```bash
python scripts/feature_engineering.py --feature-set advanced --output data/features/
```

**Model training with hyperparameter tuning:**
```bash
python scripts/train.py --tune --cv-folds 5 --n-trials 100
```

### Making Predictions

Use trained models for inference:
```bash
python scripts/predict.py --model models/segmentation_model.pkl --input new_customer_data.csv --output predictions.csv
```

## Configuration

Configuration files in the `config/` directory control various aspects of the pipeline:

### model_config.yaml
Defines model hyperparameters:
- Number of estimators
- Maximum tree depth
- Minimum samples per leaf
- Class weights for imbalanced data
- Random state for reproducibility

### preprocess_config.yaml
Specifies preprocessing operations:
- Missing value imputation strategies
- Outlier detection thresholds
- Categorical encoding methods
- Feature scaling approaches

### feature_config.yaml
Controls feature engineering:
- Temporal aggregation windows
- Feature selection criteria
- Interaction terms to generate
- Dimensionality reduction parameters

### data_schema.yaml
Documents expected data structure:
- Column names and types
- Valid value ranges
- Required fields
- Data quality constraints

## Model Information

### Customer Segmentation Model
- **Algorithm**: RandomForest Classifier
- **Purpose**: Segment customers into 5 distinct groups based on behavior and value
- **Features**: 80+ engineered features including RFM metrics, behavioral indicators, and demographic attributes
- **Performance**: 92% accuracy on validation set, 0.89 macro F1-score

### Churn Prediction Model
- **Algorithm**: RandomForest Classifier with class balancing
- **Purpose**: Predict probability of customer churn in next 90 days
- **Features**: Subset of 45 most predictive features focused on engagement trends
- **Performance**: 0.85 AUC-ROC, optimized for recall to minimize false negatives

## Data Version Control

This project uses DVC to manage large data files and model artifacts. DVC metadata files (.dvc) are tracked in Git, while actual data is stored in remote storage.

### Adding New Data Files

When adding new large data files:
```bash
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc data/.gitignore
git commit -m "Add new dataset"
dvc push
```

### Updating Data

To update existing tracked data:
```bash
# Modify the data file
dvc add data/existing_file.csv
git add data/existing_file.csv.dvc
git commit -m "Update dataset"
dvc push
```

## Development

### Adding New Features

1. Implement feature transformation in `scripts/feature_engineering.py`
2. Update feature configuration in `config/feature_config.yaml`
3. Document new features in data dictionary
4. Run pipeline to validate changes

### Model Experimentation

1. Create experiment branch
2. Modify model configuration
3. Run training with experiment tracking
4. Compare results with baseline
5. Merge successful experiments to main branch

## Maintenance

### Regular Tasks

- Monitor data quality metrics
- Retrain models monthly with new data
- Update feature engineering logic as business requirements evolve
- Review and optimize model performance
- Archive old model versions

### Troubleshooting

Common issues and solutions:

**DVC pull fails**: Check remote storage connectivity and permissions
**Memory errors during preprocessing**: Process data in chunks or increase available memory
**Model performance degradation**: Check for data drift and retrain with recent data

## Contributing

Please follow these guidelines when contributing:
- Create feature branches for new development
- Write unit tests for new functionality
- Update documentation for any changes
- Submit pull requests for review
- Ensure DVC-tracked files are properly committed

## License

This project is proprietary and confidential.

## Contact

For questions or issues, contact the data science team at ds-team@company.com