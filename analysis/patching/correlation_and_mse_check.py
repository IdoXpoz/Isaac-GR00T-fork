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
        "action_right_arm_layer_1",
        "action_right_arm_layer_3",
        "action_right_arm_layer_6",
        "action_right_arm_layer_9",
        "action_right_arm_layer_12",
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
                    cleaned = row_data.strip("[]")
                    values = np.fromstring(cleaned, sep=" ")
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
    """Calculate MSE and per-dimension correlation between each layer and layer 12."""
    reference_layer = layer_data["action_right_arm_layer_12"]
    comparison_layers = [
        "action_right_arm_layer_1",
        "action_right_arm_layer_3",
        "action_right_arm_layer_6",
        "action_right_arm_layer_9",
    ]

    results = {
        "layers": [],
        "mse_values": [],
        "correlation_values": [],
        "correlation_pvalues": [],
        "mean_correlations": [],
        "max_correlations": [],
        "max_correlation_dims": [],
        "min_correlations": [],
        "min_correlation_dims": [],
        "per_dim_correlations": [],  # Store all per-dimension correlations
    }

    print("\nCalculating metrics:")
    print("=" * 50)

    for layer_name in comparison_layers:
        layer_num = layer_name.split("_")[-1]  # Extract layer number
        layer_arrays = layer_data[layer_name]

        # Calculate MSE (average across all samples and dimensions)
        # Only use the first 7 dimensions (first action)
        mse_per_sample = []
        for i in range(len(layer_arrays)):
            if i < len(reference_layer):
                ref_action = reference_layer[i][:7] if len(reference_layer[i]) > 7 else reference_layer[i]
                layer_action = layer_arrays[i][:7] if len(layer_arrays[i]) > 7 else layer_arrays[i]
                mse = mean_squared_error(ref_action, layer_action)
                mse_per_sample.append(mse)
        avg_mse = np.mean(mse_per_sample)

        # Calculate per-dimension correlation across all samples
        min_length = min(len(layer_arrays), len(reference_layer))
        ref_subset = reference_layer[:min_length]
        layer_subset = layer_arrays[:min_length]

        # Only use the first 7 dimensions (first action)
        if len(ref_subset.shape) > 1 and ref_subset.shape[1] > 7:
            ref_subset = ref_subset[:, :7]
            layer_subset = layer_subset[:, :7]

        # Get number of dimensions
        num_dims = ref_subset.shape[1] if len(ref_subset.shape) > 1 else 1

        per_dim_correlations = []

        if len(ref_subset.shape) > 1:
            # Multi-dimensional vectors
            for dim in range(num_dims):
                ref_dim = ref_subset[:, dim]
                layer_dim = layer_subset[:, dim]
                corr, _ = pearsonr(ref_dim, layer_dim)
                if not np.isnan(corr):
                    per_dim_correlations.append(corr)
                else:
                    per_dim_correlations.append(0)
        else:
            # 1D vectors
            corr, _ = pearsonr(ref_subset, layer_subset)
            per_dim_correlations = [corr if not np.isnan(corr) else 0]

        # Calculate statistics
        mean_corr = np.mean(per_dim_correlations)
        max_corr = np.max(per_dim_correlations)
        min_corr = np.min(per_dim_correlations)
        max_dim = np.argmax(per_dim_correlations)
        min_dim = np.argmin(per_dim_correlations)

        results["layers"].append(f"Layer {layer_num}")
        results["mse_values"].append(avg_mse)
        results["correlation_values"].append(mean_corr)  # Use mean per-dimension correlation
        results["correlation_pvalues"].append(None)  # No p-value for mean of correlations
        results["mean_correlations"].append(mean_corr)
        results["max_correlations"].append(max_corr)
        results["max_correlation_dims"].append(max_dim)
        results["min_correlations"].append(min_corr)
        results["min_correlation_dims"].append(min_dim)
        results["per_dim_correlations"].append(per_dim_correlations)

        print(f"Layer {layer_num} vs Layer 12:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Mean per-dimension correlation: {mean_corr:.6f}")
        print(f"  Max per-dimension correlation: {max_corr:.6f} (dimension {max_dim})")
        print(f"  Min per-dimension correlation: {min_corr:.6f} (dimension {min_dim})")
        print(f"  Number of dimensions: {len(per_dim_correlations)}")
        print()

    return results


def calculate_cross_dataset_metrics(correct_layer_data, wrong_layer_data):
    """Calculate MSE and per-dimension correlation between wrong task layers and correct task layer 12."""
    reference_layer = correct_layer_data["action_right_arm_layer_12"]
    wrong_task_layers = [
        "action_right_arm_layer_1",
        "action_right_arm_layer_3",
        "action_right_arm_layer_6",
        "action_right_arm_layer_9",
        "action_right_arm_layer_12",
    ]

    results = {
        "layers": [],
        "mse_values": [],
        "correlation_values": [],
        "correlation_pvalues": [],
        "mean_correlations": [],
        "max_correlations": [],
        "max_correlation_dims": [],
        "min_correlations": [],
        "min_correlation_dims": [],
        "per_dim_correlations": [],  # Store all per-dimension correlations
    }

    print("\nCalculating cross-dataset metrics (Wrong Task vs Correct Task Layer 12):")
    print("=" * 70)

    for layer_name in wrong_task_layers:
        layer_num = layer_name.split("_")[-1]  # Extract layer number
        layer_arrays = wrong_layer_data[layer_name]

        # Use minimum length to avoid index errors
        min_length = min(len(layer_arrays), len(reference_layer))

        # Calculate MSE (average across all samples and dimensions)
        # Only use the first 7 dimensions (first action)
        mse_per_sample = []
        for i in range(min_length):
            ref_action = reference_layer[i][:7] if len(reference_layer[i]) > 7 else reference_layer[i]
            layer_action = layer_arrays[i][:7] if len(layer_arrays[i]) > 7 else layer_arrays[i]
            mse = mean_squared_error(ref_action, layer_action)
            mse_per_sample.append(mse)
        avg_mse = np.mean(mse_per_sample) if mse_per_sample else float("inf")

        # Calculate per-dimension correlation across all samples
        ref_subset = reference_layer[:min_length]
        layer_subset = layer_arrays[:min_length]

        # Only use the first 7 dimensions (first action)
        if len(ref_subset.shape) > 1 and ref_subset.shape[1] > 7:
            ref_subset = ref_subset[:, :7]
            layer_subset = layer_subset[:, :7]

        # Get number of dimensions
        num_dims = ref_subset.shape[1] if len(ref_subset.shape) > 1 else 1

        per_dim_correlations = []

        if len(ref_subset.shape) > 1:
            # Multi-dimensional vectors
            for dim in range(num_dims):
                ref_dim = ref_subset[:, dim]
                layer_dim = layer_subset[:, dim]
                corr, _ = pearsonr(ref_dim, layer_dim)
                if not np.isnan(corr):
                    per_dim_correlations.append(corr)
                else:
                    per_dim_correlations.append(0)
        else:
            # 1D vectors
            corr, _ = pearsonr(ref_subset, layer_subset)
            per_dim_correlations = [corr if not np.isnan(corr) else 0]

        # Calculate statistics
        if per_dim_correlations:
            mean_corr = np.mean(per_dim_correlations)
            max_corr = np.max(per_dim_correlations)
            min_corr = np.min(per_dim_correlations)
            max_dim = np.argmax(per_dim_correlations)
            min_dim = np.argmin(per_dim_correlations)
        else:
            mean_corr = max_corr = min_corr = 0
            max_dim = min_dim = -1

        results["layers"].append(f"Wrong L{layer_num}")
        results["mse_values"].append(avg_mse)
        results["correlation_values"].append(mean_corr)  # Use mean per-dimension correlation
        results["correlation_pvalues"].append(None)  # No p-value for mean of correlations
        results["mean_correlations"].append(mean_corr)
        results["max_correlations"].append(max_corr)
        results["max_correlation_dims"].append(max_dim)
        results["min_correlations"].append(min_corr)
        results["min_correlation_dims"].append(min_dim)
        results["per_dim_correlations"].append(per_dim_correlations)

        print(f"Wrong Task Layer {layer_num} vs Correct Task Layer 12:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Mean per-dimension correlation: {mean_corr:.6f}")
        if max_dim >= 0:
            print(f"  Max per-dimension correlation: {max_corr:.6f} (dimension {max_dim})")
            print(f"  Min per-dimension correlation: {min_corr:.6f} (dimension {min_dim})")
            print(f"  Number of dimensions: {len(per_dim_correlations)}")
        print()

    return results


def create_visualizations(results, cross_results=None, output_dir=None):
    """Create and display visualization plots."""
    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    if cross_results is not None:
        # Create 2x2 subplot layout: Row 1 = Correct Task, Row 2 = Wrong Task
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # ============ ROW 1: CORRECT TASK ============

        # Correct task MSE Plot (top left)
        bars1 = ax1.bar(results["layers"], results["mse_values"], color="skyblue", alpha=0.7, edgecolor="navy")
        ax1.set_title("MSE - correct task", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel("MSE", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, results["mse_values"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(results["mse_values"]) * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Correct task Per-Dimension Correlation Stats (top right)
        x = np.arange(len(results["layers"]))
        width = 0.25
        bars_mean = ax2.bar(
            x - width,
            results["mean_correlations"],
            width,
            label="Mean",
            color="gold",
            alpha=0.7,
            edgecolor="darkorange",
        )
        bars_max = ax2.bar(
            x, results["max_correlations"], width, label="Max", color="lightgreen", alpha=0.7, edgecolor="darkgreen"
        )
        bars_min = ax2.bar(
            x + width, results["min_correlations"], width, label="Min", color="salmon", alpha=0.7, edgecolor="darkred"
        )

        ax2.set_title("Per dimension correlation stats - correct task", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Correlation", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(results["layers"], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(-0.2, 1.2)

        # Add value labels on bars
        for i, (mean_bar, max_bar, min_bar) in enumerate(zip(bars_mean, bars_max, bars_min)):
            # Mean value
            mean_val = results["mean_correlations"][i]
            ax2.text(
                mean_bar.get_x() + mean_bar.get_width() / 2,
                mean_val + 0.02 if mean_val >= 0 else mean_val - 0.05,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom" if mean_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Max value
            max_val = results["max_correlations"][i]
            ax2.text(
                max_bar.get_x() + max_bar.get_width() / 2,
                max_val + 0.02 if max_val >= 0 else max_val - 0.05,
                f"{max_val:.3f}",
                ha="center",
                va="bottom" if max_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Min value
            min_val = results["min_correlations"][i]
            ax2.text(
                min_bar.get_x() + min_bar.get_width() / 2,
                min_val + 0.02 if min_val >= 0 else min_val - 0.05,
                f"{min_val:.3f}",
                ha="center",
                va="bottom" if min_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

        # ============ ROW 2: WRONG TASK ============

        # Wrong task MSE Plot (bottom left)
        bars3 = ax3.bar(
            cross_results["layers"], cross_results["mse_values"], color="lightblue", alpha=0.7, edgecolor="darkblue"
        )
        ax3.set_title("MSE - wrong task", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Wrong Task Layer", fontsize=12)
        ax3.set_ylabel("MSE", fontsize=12)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars3, cross_results["mse_values"]):
            if not np.isinf(value):  # Don't display inf values
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max([v for v in cross_results["mse_values"] if not np.isinf(v)]) * 0.01,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )

        # Wrong task Per-Dimension Correlation Stats (bottom right)
        x_cross = np.arange(len(cross_results["layers"]))
        bars_mean_cross = ax4.bar(
            x_cross - width,
            cross_results["mean_correlations"],
            width,
            label="Mean",
            color="gold",
            alpha=0.7,
            edgecolor="darkorange",
        )
        bars_max_cross = ax4.bar(
            x_cross,
            cross_results["max_correlations"],
            width,
            label="Max",
            color="lightgreen",
            alpha=0.7,
            edgecolor="darkgreen",
        )
        bars_min_cross = ax4.bar(
            x_cross + width,
            cross_results["min_correlations"],
            width,
            label="Min",
            color="salmon",
            alpha=0.7,
            edgecolor="darkred",
        )

        ax4.set_title("Per dimension correlation stats - wrong task", fontsize=14, fontweight="bold")
        ax4.set_xlabel("Wrong Task Layer", fontsize=12)
        ax4.set_ylabel("Correlation", fontsize=12)
        ax4.set_xticks(x_cross)
        ax4.set_xticklabels(cross_results["layers"], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")
        ax4.set_ylim(-0.2, 1.2)

        # Add value labels on bars
        for i, (mean_bar, max_bar, min_bar) in enumerate(zip(bars_mean_cross, bars_max_cross, bars_min_cross)):
            # Mean value
            mean_val = cross_results["mean_correlations"][i]
            ax4.text(
                mean_bar.get_x() + mean_bar.get_width() / 2,
                mean_val + 0.02 if mean_val >= 0 else mean_val - 0.05,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom" if mean_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Max value
            max_val = cross_results["max_correlations"][i]
            ax4.text(
                max_bar.get_x() + max_bar.get_width() / 2,
                max_val + 0.02 if max_val >= 0 else max_val - 0.05,
                f"{max_val:.3f}",
                ha="center",
                va="bottom" if max_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Min value
            min_val = cross_results["min_correlations"][i]
            ax4.text(
                min_bar.get_x() + min_bar.get_width() / 2,
                min_val + 0.02 if min_val >= 0 else min_val - 0.05,
                f"{min_val:.3f}",
                ha="center",
                va="bottom" if min_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

        fig.suptitle(
            "Patching analysis - comparing normal inference to actions when patching different hidden layers from the VLM to the diffusion model (correct and wrong tasks)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        if output_dir:
            output_path = os.path.join(output_dir, "layer_analysis_results.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")

        plt.show()
    else:
        # 1x2 layout for within-dataset only
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # MSE Plot
        bars1 = ax1.bar(results["layers"], results["mse_values"], color="skyblue", alpha=0.7, edgecolor="navy")
        ax1.set_title("MSE comparison", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel("MSE", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, results["mse_values"]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(results["mse_values"]) * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Per-Dimension Correlation Stats
        x = np.arange(len(results["layers"]))
        width = 0.25
        bars_mean = ax2.bar(
            x - width,
            results["mean_correlations"],
            width,
            label="Mean",
            color="gold",
            alpha=0.7,
            edgecolor="darkorange",
        )
        bars_max = ax2.bar(
            x, results["max_correlations"], width, label="Max", color="lightgreen", alpha=0.7, edgecolor="darkgreen"
        )
        bars_min = ax2.bar(
            x + width, results["min_correlations"], width, label="Min", color="salmon", alpha=0.7, edgecolor="darkred"
        )

        ax2.set_title("Per dimension correlation stats", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("Correlation", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(results["layers"], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(-0.2, 1.2)

        # Add value labels on bars
        for i, (mean_bar, max_bar, min_bar) in enumerate(zip(bars_mean, bars_max, bars_min)):
            # Mean value
            mean_val = results["mean_correlations"][i]
            ax2.text(
                mean_bar.get_x() + mean_bar.get_width() / 2,
                mean_val + 0.02 if mean_val >= 0 else mean_val - 0.05,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom" if mean_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Max value
            max_val = results["max_correlations"][i]
            ax2.text(
                max_bar.get_x() + max_bar.get_width() / 2,
                max_val + 0.02 if max_val >= 0 else max_val - 0.05,
                f"{max_val:.3f}",
                ha="center",
                va="bottom" if max_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )
            # Min value
            min_val = results["min_correlations"][i]
            ax2.text(
                min_bar.get_x() + min_bar.get_width() / 2,
                min_val + 0.02 if min_val >= 0 else min_val - 0.05,
                f"{min_val:.3f}",
                ha="center",
                va="bottom" if min_val >= 0 else "top",
                fontsize=8,
                fontweight="bold",
            )

        fig.suptitle(
            "Patching analysis - comparing normal inference to actions when patching different hidden layers from the VLM to the diffusion model",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        if output_dir:
            output_path = os.path.join(output_dir, "within_dataset_analysis.png")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
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
        print("=" * 90)
        print(
            f"{'Layer':<12} {'MSE':<15} {'Mean Corr':<12} {'Max Corr':<12} {'Max Dim':<12} {'Min Corr':<12} {'Min Dim':<12}"
        )
        print("-" * 90)
        for i, layer in enumerate(within_results["layers"]):
            print(
                f"{layer:<12} {within_results['mse_values'][i]:<15.6f} "
                f"{within_results['correlation_values'][i]:<12.6f} "
                f"{within_results['max_correlations'][i]:<12.6f} "
                f"{within_results['max_correlation_dims'][i]:<12} "
                f"{within_results['min_correlations'][i]:<12.6f} "
                f"{within_results['min_correlation_dims'][i]:<12}"
            )

        print("\n📊 CROSS-DATASET RESULTS (Wrong Task Layers vs Correct Task Layer 12)")
        print("=" * 90)
        print(
            f"{'Layer':<12} {'MSE':<15} {'Mean Corr':<12} {'Max Corr':<12} {'Max Dim':<12} {'Min Corr':<12} {'Min Dim':<12}"
        )
        print("-" * 90)
        for i, layer in enumerate(cross_results["layers"]):
            mse_str = f"{cross_results['mse_values'][i]:.6f}" if not np.isinf(cross_results["mse_values"][i]) else "inf"
            print(
                f"{layer:<12} {mse_str:<15} "
                f"{cross_results['correlation_values'][i]:<12.6f} "
                f"{cross_results['max_correlations'][i]:<12.6f} "
                f"{cross_results['max_correlation_dims'][i]:<12} "
                f"{cross_results['min_correlations'][i]:<12.6f} "
                f"{cross_results['min_correlation_dims'][i]:<12}"
            )

        # Print detailed per-dimension correlation tables
        print("\n" + "=" * 120)
        print("📊 DETAILED PER-DIMENSION CORRELATIONS - WITHIN-DATASET")
        print("=" * 120)
        for i, layer in enumerate(within_results["layers"]):
            print(f"\n{layer}:")
            print("-" * 120)
            correlations = within_results["per_dim_correlations"][i]

            # Print in rows of 8 dimensions for readability
            dims_per_row = 8
            for start_idx in range(0, len(correlations), dims_per_row):
                end_idx = min(start_idx + dims_per_row, len(correlations))
                dim_strs = []
                for dim_idx in range(start_idx, end_idx):
                    dim_strs.append(f"Dim {dim_idx:3d}: {correlations[dim_idx]:7.4f}")
                print("  " + "  ".join(dim_strs))

        print("\n" + "=" * 120)
        print("📊 DETAILED PER-DIMENSION CORRELATIONS - CROSS-DATASET")
        print("=" * 120)
        for i, layer in enumerate(cross_results["layers"]):
            print(f"\n{layer}:")
            print("-" * 120)
            correlations = cross_results["per_dim_correlations"][i]

            # Print in rows of 8 dimensions for readability
            dims_per_row = 8
            for start_idx in range(0, len(correlations), dims_per_row):
                end_idx = min(start_idx + dims_per_row, len(correlations))
                dim_strs = []
                for dim_idx in range(start_idx, end_idx):
                    dim_strs.append(f"Dim {dim_idx:3d}: {correlations[dim_idx]:7.4f}")
                print("  " + "  ".join(dim_strs))

        print("\n✅ Analysis completed successfully!")

        # Print interpretation
        print("\n💡 INTERPRETATION:")
        print("- Analysis uses only the FIRST 7 DIMENSIONS of each action vector (first action)")
        print("- All correlations are computed per-dimension (each action component independently)")
        print("- Mean Correlation = average of per-dimension correlations across all 7 dimensions")
        print("- Within-dataset results show how similar different layers are within the correct task")
        print("- Cross-dataset results show how wrong task layers compare to the correct task layer 12")
        print("- Lower correlations in cross-dataset results suggest task-specific representations")
        print("- Higher MSE in cross-dataset results indicates greater differences between tasks")
        print("- Max/Min correlations identify which specific action dimensions are most/least preserved")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
