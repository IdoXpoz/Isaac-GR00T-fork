#!/usr/bin/env python3
"""
Compare different feature types for probe evaluation.

This script compares vision_last_vector, vision_mean_pooled, text_last_vector,
and text_mean_pooled features in terms of MSE and per-dimension correlation stats.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def compare_feature_types(action_step: int = 0, show_plot: bool = True) -> Dict[str, Dict[str, float]]:
    """Create a comparison graph of MSE and per-dimension correlation stats for different feature types.

    Args:
        action_step: Which action step to compare (0-based)
        show_plot: Whether to display the plot

    Returns:
        Dictionary containing metrics for all feature types
    """
    print(f"\n📊 Comparing feature types for action step {action_step}")
    print("=" * 60)

    # Define feature types to compare
    feature_types = [
        "vision_last_vector",
        "vision_mean_pooled",
        "text_last_vector",
        "text_mean_pooled",
    ]

    # Storage for metrics
    feature_names = []
    mse_values = []
    mean_correlations = []
    max_correlations = []
    min_correlations = []
    missing_features = []

    # Load metrics for each feature type
    output_base_dir = "/home/morg/students/idoavnir/Isaac-GR00T-fork/probe/separated_embeddings"

    for feature_name in feature_types:
        probe_output_dir = os.path.join(output_base_dir, feature_name, f"action_step_{action_step}")
        metrics_path = os.path.join(probe_output_dir, "evaluation_metrics.pkl")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "rb") as f:
                    metrics = pickle.load(f)

                feature_names.append(feature_name)
                mse_values.append(metrics["mse"])

                # Get per-dimension correlations
                correlations = metrics["correlations"]
                if isinstance(correlations, list):
                    valid_corrs = [c for c in correlations if not np.isnan(c)]
                    mean_corr = np.mean(valid_corrs) if valid_corrs else 0.0
                    max_corr = np.max(valid_corrs) if valid_corrs else 0.0
                    min_corr = np.min(valid_corrs) if valid_corrs else 0.0
                else:
                    mean_corr = correlations if not np.isnan(correlations) else 0.0
                    max_corr = mean_corr
                    min_corr = mean_corr

                mean_correlations.append(mean_corr)
                max_correlations.append(max_corr)
                min_correlations.append(min_corr)

                print(f"✅ Loaded {feature_name}: MSE={metrics['mse']:.6f}, Mean Corr={mean_corr:.4f}")

            except Exception as e:
                print(f"❌ Error loading {feature_name}: {str(e)}")
                missing_features.append(feature_name)
        else:
            print(f"⚠️  Missing metrics for {feature_name}")
            missing_features.append(feature_name)

    if not feature_names:
        print("❌ No feature metrics found. Make sure to run evaluation first.")
        return {}

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different feature types
    colors = []
    for name in feature_names:
        if "vision" in name:
            if "last_vector" in name:
                colors.append("steelblue")
            else:
                colors.append("skyblue")
        else:  # text
            if "last_vector" in name:
                colors.append("darkorange")
            else:
                colors.append("orange")

    # MSE comparison
    bars1 = ax1.bar(range(len(feature_names)), mse_values, color=colors, alpha=0.7, edgecolor="navy")
    ax1.set_xlabel("Feature Type", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title(f"MSE comparison - Action Step {action_step}", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, mse_values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mse_values) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Per-dimension correlation stats
    x = np.arange(len(feature_names))
    width = 0.25

    bars_mean = ax2.bar(
        x - width,
        mean_correlations,
        width,
        label="Mean",
        color="gold",
        alpha=0.7,
        edgecolor="darkorange",
    )
    bars_max = ax2.bar(
        x,
        max_correlations,
        width,
        label="Max",
        color="lightgreen",
        alpha=0.7,
        edgecolor="darkgreen",
    )
    bars_min = ax2.bar(
        x + width,
        min_correlations,
        width,
        label="Min",
        color="salmon",
        alpha=0.7,
        edgecolor="darkred",
    )

    ax2.set_xlabel("Feature Type", fontsize=12)
    ax2.set_ylabel("Correlation", fontsize=12)
    ax2.set_title(f"Per dimension correlation stats - Action Step {action_step}", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-0.2, 1.2)

    # Add value labels on bars
    for i in range(len(feature_names)):
        # Mean value
        mean_val = mean_correlations[i]
        ax2.text(
            bars_mean[i].get_x() + bars_mean[i].get_width() / 2,
            mean_val + 0.02 if mean_val >= 0 else mean_val - 0.05,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom" if mean_val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )
        # Max value
        max_val = max_correlations[i]
        ax2.text(
            bars_max[i].get_x() + bars_max[i].get_width() / 2,
            max_val + 0.02 if max_val >= 0 else max_val - 0.05,
            f"{max_val:.3f}",
            ha="center",
            va="bottom" if max_val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )
        # Min value
        min_val = min_correlations[i]
        ax2.text(
            bars_min[i].get_x() + bars_min[i].get_width() / 2,
            min_val + 0.02 if min_val >= 0 else min_val - 0.05,
            f"{min_val:.3f}",
            ha="center",
            va="bottom" if min_val >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()

    output_dir = os.path.join(output_base_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f"feature_type_comparison_action_step_{action_step}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    # Print summary
    print(f"\n📈 Best performing feature types for action step {action_step}:")

    # Find best MSE (lowest)
    if mse_values:
        best_mse_idx = np.argmin(mse_values)
        print(f"  🏆 Lowest MSE: {feature_names[best_mse_idx]} ({mse_values[best_mse_idx]:.6f})")

    # Find best mean correlation (highest)
    if mean_correlations:
        best_corr_idx = np.argmax(mean_correlations)
        print(f"  🏆 Highest Mean Correlation: {feature_names[best_corr_idx]} ({mean_correlations[best_corr_idx]:.4f})")

    if missing_features:
        print(f"\n⚠️  Missing evaluations for: {', '.join(missing_features)}")

    print(f"\n💾 Comparison plot saved to: {plot_path}")

    # Return organized results
    results = {}
    for i, name in enumerate(feature_names):
        results[name] = {
            "mse": mse_values[i],
            "mean_correlation": mean_correlations[i],
            "max_correlation": max_correlations[i],
            "min_correlation": min_correlations[i],
        }

    return results


def main():
    """Main function to run feature type comparison."""
    import sys

    # Get action step from command line argument if provided
    action_step = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("🔍 Feature Type Comparison")
    print("=" * 60)
    print(f"Comparing: vision_last_vector, vision_mean_pooled, text_last_vector, text_mean_pooled")
    print(f"Action step: {action_step}")
    print("=" * 60)

    results = compare_feature_types(action_step=action_step, show_plot=True)

    if results:
        print("\n📊 Summary:")
        print("-" * 60)
        for feature_name, metrics in results.items():
            print(f"{feature_name}:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
            print(f"  Max Correlation: {metrics['max_correlation']:.4f}")
            print(f"  Min Correlation: {metrics['min_correlation']:.4f}")
            print()

    print("🎉 Comparison completed!")


if __name__ == "__main__":
    main()
