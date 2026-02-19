# Data Processor Application

## Description

This is a high-performance data processing utility designed to read structured data records from input files and perform statistical analysis and transformations. The application processes numerical datasets, computes aggregated metrics, and generates summary reports for downstream analytics pipelines.

## Building

To build the application, use the provided Makefile:

```
make
```

or

```
make all
```

This will compile the source code and produce the `data_processor` executable.

## Usage

Run the data processor with:

```
./data_processor
```

The application automatically reads input from `sample_data.txt` in the current directory and outputs processed results to the console.

## Purpose

The data processor is designed to handle structured numerical datasets, calculating statistics such as averages, sums, and other derived metrics. It is optimized for batch processing of data records in production environments where consistent and reliable data transformation is required.

## Input File Format

The input file should contain structured data records with numerical values. Each record is processed sequentially, and the application maintains running statistics across all records in the dataset.