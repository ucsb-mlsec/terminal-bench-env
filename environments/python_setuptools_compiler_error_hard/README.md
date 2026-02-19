# DataUtils

A Python package for data processing and transformation utilities.

## Installation

To install the package, run:

```bash
pip install .
```

## Usage

Import the package and use its utilities:

```python
import datautils

# Clean a dataset
cleaned_data = datautils.clean_data(raw_data)

# Perform statistical calculations
stats = datautils.calculate_statistics(data)

# Merge multiple datasets
merged = datautils.merge_datasets([data1, data2, data3])
```

## Features

- **Data cleaning**: Remove duplicates, handle missing values, and normalize data formats
- **Dataset merging**: Combine multiple datasets with flexible join operations
- **Statistical calculations**: Compute descriptive statistics, correlations, and distributions
- **Flexible output formatting**: Export data to CSV, JSON, or custom formats

## Requirements

- Python 3.6 or higher
- NumPy
- Pandas

## License

MIT License