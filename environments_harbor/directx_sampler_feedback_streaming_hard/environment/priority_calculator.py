#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
import os
import yaml

def load_feedback_data():
    """Load all sampler feedback CSV files"""
    csv_files = glob.glob('./streaming_data/*.csv')
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in ./streaming_data/")
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_memory_budget():
    """Load memory budget from text file"""
    with open('./memory_budget.txt', 'r') as f:
        content = f.read().strip()
        budget = float(content)
    return budget

def load_config():
    """Load configuration from YAML file"""
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def aggregate_feedback(df):
    """Aggregate sample counts across all frames"""
    # BUG: Using min instead of sum for aggregation
    aggregated = df.groupby(['texture_id', 'mip_level']).agg({
        'sample_count': 'min',  # BUG: Should be 'sum' or 'mean'
        'memory_size_mb': 'first'
    }).reset_index()
    
    return aggregated

def calculate_priorities(df, config):
    """Calculate priority scores for each texture mip level"""
    # BUG: Priority calculation is inverted - dividing by sample_count
    # This causes textures with HIGH usage to get LOW priority
    
    base_priority = config.get('base_priority', 1000.0)
    
    # BUG: Inverted formula - higher sample_count gives LOWER priority
    df['priority_score'] = base_priority / (df['sample_count'] + 1)
    
    # BUG: Not handling edge cases where sample_count might be 0 or NaN properly
    # This could cause division issues or incorrect priorities
    
    return df

def select_textures_within_budget(df, memory_budget):
    """Select textures to keep based on priority and memory budget"""
    # Sort by priority score descending (higher priority first)
    df_sorted = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
    
    cumulative_memory = 0
    selected_indices = []
    
    for idx, row in df_sorted.iterrows():
        if cumulative_memory + row['memory_size_mb'] <= memory_budget:
            cumulative_memory += row['memory_size_mb']
            selected_indices.append(idx)
        else:
            # Stop when budget exceeded
            break
    
    selected_df = df_sorted.loc[selected_indices].copy()
    total_memory = cumulative_memory
    
    return selected_df, total_memory

def calculate_correlation(df):
    """Calculate correlation between sample_count and priority_score"""
    if len(df) < 2:
        return 0.0
    
    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(df['sample_count'], df['priority_score'])
    correlation = correlation_matrix[0, 1]
    
    # Handle NaN cases
    if np.isnan(correlation):
        correlation = 0.0
    
    return correlation

def write_output(memory_used, correlation, memory_budget):
    """Write results to solution.txt"""
    # Determine test status
    if memory_used <= memory_budget and correlation > 0.8:
        status = "PASS"
    else:
        status = "FAIL"
    
    with open('./solution.txt', 'w') as f:
        f.write(f"memory_used_mb={memory_used:.1f}\n")
        f.write(f"avg_priority_correlation={correlation:.2f}\n")
        f.write(f"test_status={status}\n")

def main():
    """Main execution function"""
    # Load all data
    feedback_df = load_feedback_data()
    memory_budget = load_memory_budget()
    config = load_config()
    
    # Aggregate feedback data (with bug)
    aggregated_df = aggregate_feedback(feedback_df)
    
    # Calculate priorities (with inverted logic bug)
    priority_df = calculate_priorities(aggregated_df, config)
    
    # Select textures within memory budget
    selected_df, total_memory = select_textures_within_budget(priority_df, memory_budget)
    
    # Calculate correlation for ALL textures (not just selected)
    correlation = calculate_correlation(priority_df)
    
    # Write output
    write_output(total_memory, correlation, memory_budget)
    
    print(f"Processing complete!")
    print(f"Total textures processed: {len(priority_df)}")
    print(f"Textures selected: {len(selected_df)}")
    print(f"Memory used: {total_memory:.1f} MB")
    print(f"Memory budget: {memory_budget:.1f} MB")
    print(f"Priority correlation: {correlation:.2f}")

if __name__ == "__main__":
    main()