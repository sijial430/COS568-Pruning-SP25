import os
import re
import pandas as pd
import glob

def parse_filename(filename):
    """Extract information from the filename."""
    # Example: snip-vgg16-cifar10-singleshot-lottery-c0.1-pre0-post100.log
    basename = os.path.basename(filename)
    parts = basename.split('-')
    
    # Extract method (first part)
    method = parts[0]
    
    # Extract model
    model = parts[1]
    
    # Extract dataset
    data = parts[2]
    
    # Extract module
    module = parts[3]
    
    # Extract mode
    mode = parts[4]
    
    # Extract compression
    compression_match = re.search(r'c([\d\.]+)', basename)
    compression = compression_match.group(1) if compression_match else None
    
    # Extract pre-training epochs
    pre_match = re.search(r'pre(\d+)', basename)
    pre = pre_match.group(1) if pre_match else None
    
    # Extract post-training epochs
    post_match = re.search(r'post(\d+)', basename)
    post = post_match.group(1) if post_match else None
    
    return {
        'compression': compression,
        'model': model,
        'data': data,
        'method': method,
        'module': module,
        'mode': mode,
        'pre': pre,
        'post': post
    }

def extract_log_info(filepath):
    """Extract performance metrics from the log file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract final top1 accuracy
    accuracy_match = re.search(r'Final\s+\d+\s+[\d\.]+\s+(?:[\d\.]+(?:e[+-]\d+)?)\s+([\d\.]+)', content)
    final_top1_accuracy = accuracy_match.group(1) if accuracy_match else None
    
    # Extract post-training time
    time_match = re.search(r'Post-training time: ([\d\.]+) seconds', content)
    post_training_time = time_match.group(1) if time_match else None
    
    # Extract FLOP sparsity
    flop_match = re.search(r'FLOP Sparsity: \d+/\d+ \(([\d\.]+)\)', content)
    flop_sparsity = flop_match.group(1) if flop_match else None
    
    return {
        'final_top1_accuracy': final_top1_accuracy,
        'post_training_time': post_training_time,
        'flop_sparsity': flop_sparsity
    }

def process_log_files(directory):
    """Process all log files in the specified directory."""
    # List all log files in the directory
    log_files = glob.glob(os.path.join(directory, '*.log'))
    
    results = []
    
    for log_file in log_files:
        try:
            # Extract information from filename
            filename_info = parse_filename(log_file)
            
            # Extract information from log content
            log_info = extract_log_info(log_file)
            
            # Combine the information
            entry = {**filename_info, **log_info}
            if "grasp-vgg16-cifar10" in log_file:
                print(log_file)
                print(entry)
            results.append(entry)
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure columns are in the specified order
    desired_columns = [
        'compression', 'model', 'data', 'method', 'module', 'mode', 
        'pre', 'post', 'final_top1_accuracy', 'post_training_time', 'flop_sparsity'
    ]
    
    # Reorder columns (only include columns that exist)
    available_columns = [col for col in desired_columns if col in df.columns]
    df = df[available_columns]
    
    # Filter out rows with missing final_top1_accuracy
    if 'final_top1_accuracy' in df.columns:
        df = df.dropna(subset=['final_top1_accuracy'])
    
    # Sort the DataFrame by the specified columns
    sort_columns = ['model', 'data', 'method', 'module', 'mode', 'compression']
    sort_columns = [col for col in sort_columns if col in df.columns]
    if sort_columns:
        df = df.sort_values(by=sort_columns)
    
    return df

def main():
    # Directory containing the log files
    log_dir = "/scratch/gpfs/sl2998/workspace/COS568-Pruning-SP25/logs/"
    
    # Process the log files
    results_df = process_log_files(log_dir)
    
    # Save to CSV
    output_file = os.path.join(log_dir, "pruning_experiment_results.csv")
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    print("\nSample of the results:")
    print(results_df.head())

if __name__ == "__main__":
    main()
