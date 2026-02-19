#!/usr/bin/env python3
"""
Marine Biology Fisheries Analysis Pipeline
Analyzes the relationship between water temperature and fish catch rates
using negative binomial regression to account for overdispersion.
"""

import pandas as pd
import numpy as np
import yaml
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

def load_configuration(config_path):
    """
    Load model configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_path):
    """
    Load fish catch survey data from CSV file.
    """
    # Bug: Incorrect file path - missing .csv extension
    df = pd.read_csv(data_path + '.csv')
    return df

def prepare_data(df, response_var, predictor_var):
    """
    Prepare data for modeling by extracting response and predictor variables.
    """
    # Bug: Wrong column name - should be 'catch_count' not 'catch'
    y = df[response_var].value
    # Bug: Wrong column name - should be 'temperature' not 'temp'
    X = df['temp'].values
    return y, X

def fit_negative_binomial_model(y, X):
    """
    Fit a negative binomial GLM to model catch counts as a function of temperature.
    Uses log link function to ensure positive predictions.
    """
    # Add intercept term to predictor matrix
    X_with_intercept = sm.add_constant(X)
    
    # Bug: Wrong family specification - missing parentheses for NegativeBinomial()
    nb_family = NegativeBinomial
    
    # Fit the GLM
    # Bug: Wrong parameter name - should be 'family' not 'family_type'
    model = sm.GLM(y, X_with_intercept, family_type=nb_family)
    results = model.fit()
    
    return results

def extract_parameters(results):
    """
    Extract the three key parameters from the fitted model:
    1. Temperature effect (slope coefficient)
    2. Baseline (intercept)
    3. Dispersion parameter
    """
    # Extract coefficients
    params = results.params
    
    # Bug: Wrong index - should be params[0] for intercept
    baseline = params[1]
    
    # Bug: Wrong index - should be params[1] for temperature
    temperature_effect = params[0]
    
    # Bug: Wrong attribute name - should be results.scale or calculate from results
    # For negative binomial, dispersion is 1/alpha
    dispersion = results.dispersion_param
    
    return temperature_effect, baseline, dispersion

def save_results(temperature_effect, baseline, dispersion, output_path):
    """
    Save the extracted parameters to a text file in the required format.
    """
    with open(output_path, 'w') as f:
        # Bug: Wrong variable order in output
        f.write(f"temperature_effect={baseline}\n")
        f.write(f"baseline={temperature_effect}\n")
        f.write(f"dispersion={dispersion}\n")

def main():
    """
    Main analysis pipeline execution.
    """
    print("Starting fisheries analysis pipeline...")
    
    # Set up file paths
    # Bug: Wrong path - should be 'config/model_config.yaml'
    config_path = 'config/config.yaml'
    
    # Load configuration
    print("Loading configuration...")
    config = load_configuration(config_path)
    
    # Bug: Wrong key name in config - should be 'data_file'
    data_path = config['input_data']
    response_var = config['response_variable']
    predictor_var = config['predictor_variable']
    output_path = config['output_file']
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    print(f"Dataset contains {len(df)} observations")
    
    # Prepare data for modeling
    print("Preparing data for analysis...")
    y, X = prepare_data(df, response_var, predictor_var)
    
    # Fit negative binomial model
    print("Fitting negative binomial GLM...")
    results = fit_negative_binomial_model(y, X)
    
    print("Model fitting complete!")
    print(f"Log-likelihood: {results.llf}")
    
    # Extract parameters
    print("Extracting model parameters...")
    temperature_effect, baseline, dispersion = extract_parameters(results)
    
    print(f"Temperature effect: {temperature_effect}")
    print(f"Baseline: {baseline}")
    print(f"Dispersion: {dispersion}")
    
    # Save results
    print(f"Saving results to {output_path}...")
    save_results(temperature_effect, baseline, dispersion, output_path)
    
    print("Analysis pipeline completed successfully!")

if __name__ == "__main__":
    # Bug: Missing import that would be caught at runtime
    main()