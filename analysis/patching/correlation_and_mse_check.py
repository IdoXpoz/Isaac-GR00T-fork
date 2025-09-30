#!/usr/bin/env python3
"""
Analysis script for computing MSE and correlation between different VLM layer outputs.
Compares layers 1, 3, 6, 9 against layer 12 (reference layer) within the same dataset,
and also compares all layers from wrong task dataset against layer 12 from correct task dataset.
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

def calculate_cross_dataset_metrics(correct_layer_data, wrong_layer_data):
    """Calculate MSE and correlation between wrong task layers and correct task layer 12."""
    reference_layer = correct_layer_data['action_right_arm_layer_12']
    wrong_task_layers = [
        'action_right_arm_layer_1',
        'action_right_arm_layer_3', 
        'action_right_arm_layer_6',
        'action_right_arm_layer_9',
        'action_right_arm_layer_12'
    ]
    
    results = {
        'layers': [],
        'mse_values': [],
        'correlation_values': [],
        'correlation_pvalues': []
    }
    
    print("\nCalculating cross-dataset metrics (Wrong Task vs Correct Task Layer 12):")
    print("=" * 70)
    
    for layer_name in wrong_task_layers:
        layer_num = layer_name.split('_')[-1]  # Extract layer number
        layer_arrays = wrong_layer_data[layer_name]
        
        # Calculate MSE for each sample pair, then average
        mse_per_sample = []
        correlation_per_sample = []
        
        # Use minimum length to avoid index errors
        min_length = min(len(layer_arrays), len(reference_layer))
        
        for i in range(min_length):
            # MSE between corresponding samples
            mse = mean_squared_error(reference_layer[i], layer_arrays[i])
            mse_per_sample.append(mse)
            
            # Correlation between corresponding samples
            corr, _ = pearsonr(reference_layer[i].flatten(), layer_arrays[i].flatten())
            if not np.isnan(corr):
                correlation_per_sample.append(corr)
        
        avg_mse = np.mean(mse_per_sample) if mse_per_sample else float('inf')
        avg_correlation = np.mean(correlation_per_sample) if correlation_per_sample else 0
        
        # Also calculate overall correlation (flattening all data)
        flat_ref = reference_layer[:min_length].flatten()
        flat_layer = layer_arrays[:min_length].flatten()
        overall_corr, p_value = pearsonr(flat_ref, flat_layer)
        
        results['layers'].append(f"Wrong L{layer_num}")
        results['mse_values'].append(avg_mse)
        results['correlation_values'].append(overall_corr)
        results['correlation_pvalues'].append(p_value)
        
        print(f"Wrong Task Layer {layer_num} vs Correct Task Layer 12:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Overall Correlation: {overall_corr:.6f} (p={p_value:.2e})")
        print(f"  Average per-sample correlation: {avg_correlation:.6f}")
        print()
    
    return results

def create_visualizations(results, cross_results=None, output_dir=None):
    """Create and display visualization plots."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    if cross_results is not None:
        # Create 2x2 subplot layout for both comparisons
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Within-dataset MSE Plot (top left)
        bars1 = ax1.bar(results['layers'], results['mse_values'], 
                        color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_title('Within-Dataset: MSE vs Correct Layer 12', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, results['mse_values']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(results['mse_values'])*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Within-dataset Correlation Plot (top right)
        bars2 = ax2.bar(results['layers'], results['correlation_values'], 
                        color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_title('Within-Dataset: Correlation with Correct Layer 12', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Pearson Correlation', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)  # Correlation should be between 0 and 1 for this context
        
        # Add value labels on bars
        for bar, value in zip(bars2, results['correlation_values']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Cross-dataset MSE Plot (bottom left)
        bars3 = ax3.bar(cross_results['layers'], cross_results['mse_values'], 
                        color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax3.set_title('Cross-Dataset: Wrong Task Layers vs Correct Layer 12', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Wrong Task Layer', fontsize=12)
        ax3.set_ylabel('MSE', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars3, cross_results['mse_values']):
            if not np.isinf(value):  # Don't display inf values
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max([v for v in cross_results['mse_values'] if not np.isinf(v)])*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Cross-dataset Correlation Plot (bottom right)
        bars4 = ax4.bar(cross_results['layers'], cross_results['correlation_values'], 
                        color='orange', alpha=0.7, edgecolor='darkorange')
        ax4.set_title('Cross-Dataset: Wrong Task Layers vs Correct Layer 12', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Wrong Task Layer', fontsize=12)
        ax4.set_ylabel('Pearson Correlation', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        # Don't set ylim for cross-dataset correlations as they might be negative
        
        # Add value labels on bars
        for bar, value in zip(bars4, cross_results['correlation_values']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if value >= 0 else bar.get_height() - 0.05,
                    f'{value:.4f}', ha='center', va='bottom' if value >= 0 else 'top', 
                    fontweight='bold', fontsize=10)
        
        plt.suptitle('VLM Layer Analysis: Within-Dataset vs Cross-Dataset Comparisons', fontsize=16, fontweight='bold')
    else:
        # Original 1x2 layout for within-dataset only
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
    correct_data_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/action_different_vlm_layers_data"
    wrong_data_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/action_wrong_task_different_vlm_layers_data"
    output_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/analysis/patching"
    
    try:
        # Load correct task data
        print("🔄 Loading correct task parquet data...")
        correct_df = load_parquet_data(correct_data_dir)
        
        # Load wrong task data
        print("🔄 Loading wrong task parquet data...")
        wrong_df = load_parquet_data(wrong_data_dir)
        
        # Extract layer data from both datasets
        print("\n🔄 Extracting correct task layer data...")
        correct_layer_data = extract_layer_data(correct_df)
        
        print("\n🔄 Extracting wrong task layer data...")
        wrong_layer_data = extract_layer_data(wrong_df)
        
        # Calculate within-dataset metrics (correct task layers vs correct task layer 12)
        print("\n🔄 Calculating within-dataset MSE and correlation metrics...")
        within_results = calculate_metrics(correct_layer_data)
        
        # Calculate cross-dataset metrics (wrong task layers vs correct task layer 12)
        print("\n🔄 Calculating cross-dataset MSE and correlation metrics...")
        cross_results = calculate_cross_dataset_metrics(correct_layer_data, wrong_layer_data)
        
        # Create visualizations
        print("\n🔄 Creating visualizations...")
        create_visualizations(within_results, cross_results, output_dir)
        
        # Print summary tables
        print("\n📊 WITHIN-DATASET RESULTS (Correct Task Layers vs Correct Task Layer 12)")
        print("=" * 80)
        print(f"{'Layer':<12} {'MSE':<15} {'Correlation':<15} {'P-value':<10}")
        print("-" * 80)
        for i, layer in enumerate(within_results['layers']):
            print(f"{layer:<12} {within_results['mse_values'][i]:<15.6f} "
                  f"{within_results['correlation_values'][i]:<15.6f} "
                  f"{within_results['correlation_pvalues'][i]:<10.2e}")
        
        print("\n📊 CROSS-DATASET RESULTS (Wrong Task Layers vs Correct Task Layer 12)")
        print("=" * 80)
        print(f"{'Layer':<12} {'MSE':<15} {'Correlation':<15} {'P-value':<10}")
        print("-" * 80)
        for i, layer in enumerate(cross_results['layers']):
            mse_str = f"{cross_results['mse_values'][i]:.6f}" if not np.isinf(cross_results['mse_values'][i]) else "inf"
            print(f"{layer:<12} {mse_str:<15} "
                  f"{cross_results['correlation_values'][i]:<15.6f} "
                  f"{cross_results['correlation_pvalues'][i]:<10.2e}")
        
        print("\n✅ Analysis completed successfully!")
        
        # Print interpretation
        print("\n💡 INTERPRETATION:")
        print("- Within-dataset results show how similar different layers are within the correct task")
        print("- Cross-dataset results show how wrong task layers compare to the correct task layer 12")
        print("- Lower correlations in cross-dataset results suggest task-specific representations")
        print("- Higher MSE in cross-dataset results indicates greater differences between tasks")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
