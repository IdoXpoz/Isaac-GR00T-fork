#!/usr/bin/env python3
"""
Analysis script for computing MSE and correlation between different VLM layer outputs.
Compares layers 1, 3, 6, 9 against layer 12 (reference layer).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os

def load_parquet_data(data_dir):
    """Load the merged parquet file containing VLM layer outputs."""
    parquet_path = os.path.join(data_dir, "batches_parquet", "merged_batches.parquet")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Merged parquet file not found at: {parquet_path}")
    
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    return df

def extract_layer_data(df):
    """Extract and convert layer data from string format to numpy arrays."""
    layer_columns = [
        'action_right_arm_layer_1',
        'action_right_arm_layer_3', 
        'action_right_arm_layer_6',
        'action_right_arm_layer_9',
        'action_right_arm_layer_12'
    ]
    
    layer_data = {}
    
    for col in layer_columns:
        print(f"Processing {col}...")
        
        # Convert string representations to numpy arrays
        layer_arrays = []
        for i, row_data in enumerate(df[col]):
            if isinstance(row_data, str):
                # Parse string representation of numpy array
                try:
                    # Remove brackets and split by whitespace
                    cleaned = row_data.strip('[]')
                    values = np.fromstring(cleaned, sep=' ')
                    layer_arrays.append(values)
                except:
                    print(f"Error parsing row {i} for {col}")
                    continue
            elif isinstance(row_data, np.ndarray):
                layer_arrays.append(row_data)
            else:
                print(f"Unexpected data type for {col}: {type(row_data)}")
                continue
        
        layer_data[col] = np.array(layer_arrays)
        print(f"  Shape: {layer_data[col].shape}")
    
    return layer_data

def calculate_metrics(layer_data):
    """Calculate MSE and correlation between each layer and layer 12."""
    reference_layer = layer_data['action_right_arm_layer_12']
    comparison_layers = [
        'action_right_arm_layer_1',
        'action_right_arm_layer_3', 
        'action_right_arm_layer_6',
        'action_right_arm_layer_9'
    ]
    
    results = {
        'layers': [],
        'mse_values': [],
        'correlation_values': [],
        'correlation_pvalues': []
    }
    
    print("\nCalculating metrics:")
    print("=" * 50)
    
    for layer_name in comparison_layers:
        layer_num = layer_name.split('_')[-1]  # Extract layer number
        layer_arrays = layer_data[layer_name]
        
        # Calculate MSE for each sample pair, then average
        mse_per_sample = []
        correlation_per_sample = []
        
        for i in range(len(layer_arrays)):
            if i < len(reference_layer):
                # MSE between corresponding samples
                mse = mean_squared_error(reference_layer[i], layer_arrays[i])
                mse_per_sample.append(mse)
                
                # Correlation between corresponding samples
                corr, _ = pearsonr(reference_layer[i].flatten(), layer_arrays[i].flatten())
                if not np.isnan(corr):
                    correlation_per_sample.append(corr)
        
        avg_mse = np.mean(mse_per_sample)
        avg_correlation = np.mean(correlation_per_sample)
        
        # Also calculate overall correlation (flattening all data)
        flat_ref = reference_layer[:len(layer_arrays)].flatten()
        flat_layer = layer_arrays.flatten()
        overall_corr, p_value = pearsonr(flat_ref, flat_layer)
        
        results['layers'].append(f"Layer {layer_num}")
        results['mse_values'].append(avg_mse)
        results['correlation_values'].append(overall_corr)
        results['correlation_pvalues'].append(p_value)
        
        print(f"Layer {layer_num} vs Layer 12:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Overall Correlation: {overall_corr:.6f} (p={p_value:.2e})")
        print(f"  Average per-sample correlation: {avg_correlation:.6f}")
        print()
    
    return results

def create_visualizations(results, output_dir=None):
    """Create and display visualization plots."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE Plot
    bars1 = ax1.bar(results['layers'], results['mse_values'], 
                    color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Mean Squared Error vs Layer 12', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, results['mse_values']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(results['mse_values'])*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Correlation Plot
    bars2 = ax2.bar(results['layers'], results['correlation_values'], 
                    color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_title('Correlation with Layer 12', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)  # Correlation should be between 0 and 1 for this context
    
    # Add value labels on bars
    for bar, value in zip(bars2, results['correlation_values']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = os.path.join(output_dir, 'layer_analysis_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    """Main analysis function."""
    # Define paths
    data_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/action_different_vlm_layers_data"
    output_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/analysis/patching"
    
    try:
        # Load data
        print("🔄 Loading parquet data...")
        df = load_parquet_data(data_dir)
        
        # Extract layer data
        print("\n🔄 Extracting layer data...")
        layer_data = extract_layer_data(df)
        
        # Calculate metrics
        print("\n🔄 Calculating MSE and correlation metrics...")
        results = calculate_metrics(layer_data)
        
        # Create visualizations
        print("\n🔄 Creating visualizations...")
        create_visualizations(results, output_dir)
        
        # Print summary table
        print("\n📊 SUMMARY RESULTS")
        print("=" * 60)
        print(f"{'Layer':<10} {'MSE':<15} {'Correlation':<15} {'P-value':<10}")
        print("-" * 60)
        for i, layer in enumerate(results['layers']):
            print(f"{layer:<10} {results['mse_values'][i]:<15.6f} "
                  f"{results['correlation_values'][i]:<15.6f} "
                  f"{results['correlation_pvalues'][i]:<10.2e}")
        
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
