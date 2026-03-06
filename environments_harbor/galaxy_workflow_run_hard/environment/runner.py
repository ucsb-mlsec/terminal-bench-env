#!/usr/bin/env python3

import os
import sys
import glob
import configparser
# Missing import: yaml
# Missing import: pandas

def load_config():
    """Load configuration from config.ini file"""
    config = configparser.ConfigParser()
    # Bug: Wrong path - should use os.path.dirname(__file__)
    config_path = '/etc/workflow-engine/config.ini'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    
    config.read(config_path)
    return config

def parse_workflow(workflow_path):
    """Parse YAML workflow definition"""
    if not os.path.exists(workflow_path):
        print(f"Error: Workflow file not found at {workflow_path}")
        sys.exit(1)
    
    with open(workflow_path, 'r') as f:
        # Bug: yaml module not imported
        workflow = yaml.safe_load(f)
    
    return workflow

def process_filter_stage(input_files, stage_config, output_dir):
    """Filter CSV rows based on value thresholds"""
    print(f"Executing filter stage: {stage_config.get('name', 'unnamed')}")
    
    threshold = stage_config.get('threshold', 0)
    operator = stage_config.get('operator', '>')
    
    filtered_data = []
    total_rows = 0
    
    for input_file in input_files:
        print(f"  Processing {input_file}...")
        # Bug: pandas not imported
        df = pd.read_csv(input_file)
        
        # Apply filter based on operator
        if operator == '>':
            filtered_df = df[df['value'] > threshold]
        elif operator == '<':
            filtered_df = df[df['value'] < threshold]
        elif operator == '>=':
            filtered_df = df[df['value'] >= threshold]
        elif operator == '<=':
            filtered_df = df[df['value'] <= threshold]
        elif operator == '==':
            filtered_df = df[df['value'] == threshold]
        else:
            filtered_df = df
        
        total_rows += len(filtered_df)
        filtered_data.append(filtered_df)
        
        # Save filtered output
        output_filename = os.path.basename(input_file).replace('.csv', '_filtered.csv')
        output_path = os.path.join(output_dir, output_filename)
        # Bug: Wrong method name - should be to_csv
        filtered_df.save_csv(output_path, index=False)
    
    return filtered_data, total_rows

def process_aggregate_stage(input_data, stage_config, output_dir):
    """Group by category and compute statistics"""
    print(f"Executing aggregate stage: {stage_config.get('name', 'unnamed')}")
    
    # Combine all input dataframes
    combined_df = pd.concat(input_data, ignore_index=True)
    
    # Group by category and compute statistics
    grouped = combined_df.groupby('category').agg({
        'value': ['count', 'mean', 'sum', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['category', 'count', 'mean', 'sum', 'min', 'max']
    
    # Save aggregated results
    output_path = os.path.join(output_dir, 'aggregated_results.csv')
    grouped.to_csv(output_path, index=False)
    
    print(f"  Aggregated data saved to {output_path}")
    
    return grouped

def process_report_stage(aggregated_data, stage_config, output_dir, total_rows, files_processed):
    """Generate summary report"""
    print(f"Executing report stage: {stage_config.get('name', 'unnamed')}")
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("DATA PROCESSING PIPELINE REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    report_lines.append(f"Files Processed: {files_processed}")
    report_lines.append(f"Total Rows (after filtering): {total_rows}")
    report_lines.append("")
    report_lines.append("CATEGORY STATISTICS:")
    report_lines.append("-"*60)
    
    for _, row in aggregated_data.iterrows():
        report_lines.append(f"\nCategory: {row['category']}")
        report_lines.append(f"  Count: {row['count']}")
        report_lines.append(f"  Mean: {row['mean']:.2f}")
        report_lines.append(f"  Sum: {row['sum']:.2f}")
        report_lines.append(f"  Min: {row['min']:.2f}")
        report_lines.append(f"  Max: {row['max']:.2f}")
    
    report_lines.append("")
    report_lines.append("="*60)
    
    # Save report
    output_path = os.path.join(output_dir, 'summary_report.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Report saved to {output_path}")
    
    return output_path

def execute_stage(stage, input_files, current_data, output_dir):
    """Execute a single pipeline stage"""
    stage_type = stage.get('type')
    
    if stage_type == 'filter':
        filtered_data, total_rows = process_filter_stage(input_files, stage, output_dir)
        return filtered_data, total_rows
    elif stage_type == 'aggregate':
        aggregated_data = process_aggregate_stage(current_data, stage, output_dir)
        return aggregated_data, None
    elif stage_type == 'report':
        # This stage needs additional context
        return None, None
    else:
        print(f"Warning: Unknown stage type '{stage_type}'")
        return current_data, None

def main():
    """Main workflow execution function"""
    if len(sys.argv) < 2:
        print("Usage: runner.py <workflow_yaml_path>")
        sys.exit(1)
    
    workflow_path = sys.argv[1]
    
    print("="*60)
    print("WORKFLOW ENGINE STARTING")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Parse workflow
    workflow = parse_workflow(workflow_path)
    
    # Get input and output directories from config
    # Bug: Wrong section name - should match config.ini
    input_dir = config.get('paths', 'input_directory')
    output_dir = config.get('paths', 'output_directory')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input files
    input_pattern = os.path.join(input_dir, '*.csv')
    input_files = sorted(glob.glob(input_pattern))
    
    if not input_files:
        print(f"Error: No CSV files found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(input_files)} input files")
    
    # Execute pipeline stages
    stages = workflow.get('pipeline', {}).get('stages', [])
    
    if not stages:
        print("Error: No stages defined in workflow")
        sys.exit(1)
    
    print(f"Pipeline has {len(stages)} stages\n")
    
    current_data = None
    total_rows = 0
    files_processed = len(input_files)
    
    for i, stage in enumerate(stages):
        print(f"\n--- Stage {i+1}/{len(stages)} ---")
        
        if stage.get('type') == 'filter':
            current_data, total_rows = execute_stage(stage, input_files, current_data, output_dir)
        elif stage.get('type') == 'aggregate':
            current_data, _ = execute_stage(stage, input_files, current_data, output_dir)
        elif stage.get('type') == 'report':
            process_report_stage(current_data, stage, output_dir, total_rows, files_processed)
        else:
            current_data, _ = execute_stage(stage, input_files, current_data, output_dir)
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Write status file
    status_dir = '/solution'
    os.makedirs(status_dir, exist_ok=True)
    status_file = os.path.join(status_dir, 'workflow_status.txt')
    
    with open(status_file, 'w') as f:
        f.write(f"status=SUCCESS\n")
        f.write(f"files_processed={files_processed}\n")
        f.write(f"total_rows={total_rows}\n")
    
    print(f"\nStatus written to {status_file}")

if __name__ == '__main__':
    main()