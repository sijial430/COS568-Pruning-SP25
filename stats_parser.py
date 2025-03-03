# import pickle

# file_path = '/home/sl2998/workspace/COS568-Pruning-SP25/Results/data/singleshot/grasp-vgg16-cifar10-singleshot-lottery-c0.5-pre0-post100/compression.pkl'

# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# print(data[data['param']=='weight'][['module', 'sparsity', 'flops']])

import os
import pickle
import pandas as pd
import glob
from collections import defaultdict

def parse_compression_files(base_dir='/home/sl2998/workspace/COS568-Pruning-SP25/Results/data/'):
    """
    Parse all compression.pkl files under the singleshot directory that match
    'vgg16-cifar10-singleshot-lottery-c0.5' pattern and organize by experiment name.
    
    Args:
        base_dir: Base directory where the singleshot directory is located
    
    Returns:
        Two DataFrames:
        - DataFrame summarizing weight sparsity by experiment and module
        - DataFrame summarizing actual FLOPs (considering sparsity) by experiment and module
    """
    # Pattern to search for matching files
    pattern = os.path.join(base_dir, 'singleshot', '*-vgg16-cifar10-singleshot-lottery-c0.5-*', 'compression.pkl')
    
    # Find all matching files
    matching_files = glob.glob(pattern)
    
    print(f"Found {len(matching_files)} matching files")
    
    # Dictionary to store results by experiment name
    sparsity_by_experiment = defaultdict(list)
    total_flops_by_experiment = defaultdict(dict)
    actual_flops_by_experiment = defaultdict(dict)
    
    # Process each file
    for file_path in matching_files:
        # Extract experiment name from the path
        # The path format should be: .../singleshot/EXPERIMENT-vgg16-cifar10-singleshot-lottery-c0.5-.../compression.pkl
        dir_name = os.path.basename(os.path.dirname(file_path))
        
        # Extract first word from the experiment name
        experiment_name = dir_name.split('-')[0]
        
        try:
            # Load the pickle file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            
            # Debug: Print some sample data to see structure
            print(f"\nSample data from {file_path}:")
            print(data[['module', 'param', 'sparsity', 'flops']].head(5))
            
            # Create a dictionary to store total FLOPs per module
            module_total_flops = {}
            
            # Create a dictionary to store sparsity values per module for weight params
            module_weight_sparsity = {}
            
            # First, calculate total FLOPs and extract weight sparsity for each module
            for _, row in data.iterrows():
                module = row['module']
                param_type = row['param']
                
                # Accumulate total FLOPs
                if module not in module_total_flops:
                    module_total_flops[module] = 0
                module_total_flops[module] += row['flops']
                
                # Store weight sparsity
                if param_type == 'weight':
                    module_weight_sparsity[module] = row['sparsity']
                    
                    # Add to sparsity results
                    sparsity_by_experiment[experiment_name].append({
                        'experiment': experiment_name,
                        'full_path': file_path,
                        'module': module,
                        'sparsity': row['sparsity']
                    })
            
            # Store the total FLOPs for each module
            total_flops_by_experiment[experiment_name] = module_total_flops
            
            # Calculate actual FLOPs considering sparsity for each module
            for module, total_flops in module_total_flops.items():
                if module in module_weight_sparsity:
                    # Actual FLOPs = Total FLOPs * (1 - weight sparsity)
                    # Higher sparsity means fewer operations
                    sparsity = module_weight_sparsity[module]
                    actual_flops = int(total_flops * sparsity)
                    actual_flops_by_experiment[experiment_name][module] = actual_flops
                else:
                    # If no sparsity info, assume no pruning
                    actual_flops_by_experiment[experiment_name][module] = total_flops
            
            print(f"Processed: {file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Process sparsity data to create a DataFrame
    sparsity_df = process_data_with_preserved_order(sparsity_by_experiment)
    
    # Process actual FLOPs data to create a DataFrame
    actual_flops_df = process_dict_data(actual_flops_by_experiment, sparsity_by_experiment)
    
    # Process total FLOPs data (before applying sparsity)
    total_flops_df = process_dict_data(total_flops_by_experiment, sparsity_by_experiment)
    
    return sparsity_df, actual_flops_df, total_flops_df

def process_data_with_preserved_order(data_by_experiment):
    """
    Process data organized as a list of dictionaries per experiment to create a DataFrame 
    with the original module order preserved.
    
    Args:
        data_by_experiment: Dictionary with data by experiment
    
    Returns:
        DataFrame with modules as rows and experiments as columns
    """
    # Convert results to DataFrame
    all_results = []
    for experiment, results in data_by_experiment.items():
        all_results.extend(results)
    
    if not all_results:
        print("No data found in matching files")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # To preserve the original module order, we need to handle this differently
    # First, get a reference ordering from the first file processed
    if len(data_by_experiment) > 0:
        first_exp = list(data_by_experiment.keys())[0]
        module_order = [item['module'] for item in data_by_experiment[first_exp]]
        
        # Create a custom ordered pivot table
        pivot_data = {}
        pivot_data['module'] = module_order
        
        for experiment in data_by_experiment.keys():
            # Create a mapping from module to value for this experiment
            module_to_value = {item['module']: item['sparsity'] 
                             for item in data_by_experiment[experiment]}
            
            # Add column for this experiment, preserving module order
            pivot_data[experiment] = [module_to_value.get(module, float('nan')) 
                                    for module in module_order]
        
        pivot_df = pd.DataFrame(pivot_data)
    else:
        # Fallback to regular pivot if no data
        pivot_df = results_df.pivot_table(
            index='module', 
            columns='experiment',
            values='sparsity',
            aggfunc='mean'
        ).reset_index()
    
    return pivot_df

def process_dict_data(data_by_experiment, sparsity_by_experiment):
    """
    Process data organized as nested dictionaries to create a DataFrame 
    with the original module order preserved.
    
    Args:
        data_by_experiment: Dictionary with data by experiment and module
        sparsity_by_experiment: Dictionary with sparsity data for module ordering
    
    Returns:
        DataFrame with modules as rows and experiments as columns
    """
    if not data_by_experiment:
        print("No data found in matching files")
        return pd.DataFrame()
    
    # Get module ordering from the sparsity data to maintain consistency
    if len(sparsity_by_experiment) > 0:
        first_exp = list(sparsity_by_experiment.keys())[0]
        module_order = [item['module'] for item in sparsity_by_experiment[first_exp]]
        
        # Create a custom ordered table
        pivot_data = {}
        pivot_data['module'] = module_order
        
        for experiment in data_by_experiment.keys():
            # Get the data for this experiment
            module_to_value = data_by_experiment[experiment]
            
            # Add column for this experiment, preserving module order
            pivot_data[experiment] = [module_to_value.get(module, float('nan')) 
                                    for module in module_order]
        
        pivot_df = pd.DataFrame(pivot_data)
    else:
        # If no sparsity data for ordering, create DataFrame from data directly
        # This won't preserve any specific order
        flattened_data = []
        for experiment, module_data in data_by_experiment.items():
            for module, value in module_data.items():
                flattened_data.append({
                    'experiment': experiment,
                    'module': module,
                    'value': value
                })
        
        results_df = pd.DataFrame(flattened_data)
        pivot_df = results_df.pivot_table(
            index='module', 
            columns='experiment',
            values='value',
            aggfunc='mean'
        ).reset_index()
    
    return pivot_df

if __name__ == "__main__":
    # Run the parser
    sparsity_table, actual_flops_table, total_flops_table = parse_compression_files()
    
    if not sparsity_table.empty:
        print("\nWeight Sparsity by Experiment and Module:")
        print(sparsity_table)
        
        # Save results to CSV
        sparsity_file = "weight_sparsity_summary.csv"
        sparsity_table.to_csv(sparsity_file, index=False)
        print(f"\nSparsity results saved to {sparsity_file}")
    else:
        print("No sparsity results to display")
        
    if not actual_flops_table.empty:
        print("\nActual FLOPs (with sparsity) by Experiment and Module:")
        print(actual_flops_table)
        
        # Save results to CSV
        actual_flops_file = "actual_flops_summary.csv"
        actual_flops_table.to_csv(actual_flops_file, index=False)
        print(f"\nActual FLOPs results saved to {actual_flops_file}")
    else:
        print("No actual FLOPs results to display")
        
    if not total_flops_table.empty:
        print("\nTotal FLOPs (before sparsity) by Experiment and Module:")
        print(total_flops_table)
        
        # Save results to CSV
        total_flops_file = "total_flops_summary.csv"
        total_flops_table.to_csv(total_flops_file, index=False)
        print(f"\nTotal FLOPs results saved to {total_flops_file}")
    else:
        print("No total FLOPs results to display")