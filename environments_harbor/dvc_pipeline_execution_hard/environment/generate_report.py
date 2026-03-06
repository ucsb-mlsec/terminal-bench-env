#!/usr/bin/env python3

import pandas as pd
import yaml

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

reporting_params = params['reporting']

# Read aggregated data
df = pd.read_csv('data/aggregated_data.csv')

# Select output columns
output_cols = reporting_params['output_columns']
df = df[output_cols]

# Sort data - BUG: incorrect sort order logic (reversed)
sort_by = reporting_params['sort_by']
sort_order = reporting_params['sort_order']

# BUG: The logic is inverted - 'ascending' results in descending sort
if sort_order == 'ascending':
    df = df.sort_values(by=sort_by, ascending=False)
else:
    df = df.sort_values(by=sort_by, ascending=True)

# Format revenue to 2 decimal places
if 'revenue' in df.columns:
    df['revenue'] = df['revenue'].round(2)

# Save report
df.to_csv('data/report.csv', index=False)

print("Report generated successfully")