#!/usr/bin/env python3
"""
Visualize layer comparison for correct and wrong task probes.

Creates a 2x2 grid comparing MSE and per-dimension correlation stats
for different VLM hidden layers in both correct and wrong task scenarios.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from probe_comparison_data import correct_task_data, wrong_task_data


def visualize_layer_comparison(show_plot: bool = True, output_dir: Optional[str] = None):
    """Create 2x2 comparison plot of correct vs wrong task across different layers.

    Args:
        show_plot: Whether to display the plot
        output_dir: Directory to save the plot (defaults to current directory)
    """
    print("\n📊 Creating layer comparison visualization...")
    print("=" * 60)

    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

    # ============ PROCESS CORRECT TASK DATA ============
    correct_probe_names = []
    correct_mse_values = []
    correct_mean_corrs = []

    # Separate mean_pooled and last_vector for coloring
    for probe_name in sorted(correct_task_data.keys()):
        correct_probe_names.append(probe_name)
        correct_mse_values.append(correct_task_data[probe_name]["mse"])
        correct_mean_corrs.append(correct_task_data[probe_name]["mean_correlation"])

    # Colors for correct task
    correct_colors = []
    for name in correct_probe_names:
        if "mean_pooled" in name:
            correct_colors.append("steelblue")
        else:
            correct_colors.append("darkorange")

    # ============ ROW 1: CORRECT TASK ============

    # Correct task MSE (top left)
    bars1 = ax1.bar(
        range(len(correct_probe_names)), correct_mse_values, color=correct_colors, alpha=0.7, edgecolor="navy"
    )
    ax1.set_title("MSE - correct task", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(correct_probe_names)))
    ax1.set_xticklabels(correct_probe_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.14)  # Increase upper limit to give space for legend

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, correct_mse_values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(correct_mse_values) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Correct task Mean Correlation (top right)
    bars2 = ax2.bar(
        range(len(correct_probe_names)), correct_mean_corrs, color=correct_colors, alpha=0.7, edgecolor="navy"
    )
    ax2.set_title("Per engine correlation stats - correct task", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(len(correct_probe_names)))
    ax2.set_xticklabels(correct_probe_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 0.92)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, correct_mean_corrs)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # ============ PROCESS WRONG TASK DATA ============
    wrong_probe_names = []
    wrong_mse_values = []
    wrong_mean_corrs = []

    for probe_name in sorted(wrong_task_data.keys()):
        wrong_probe_names.append(probe_name)
        wrong_mse_values.append(wrong_task_data[probe_name]["mse"])
        wrong_mean_corrs.append(wrong_task_data[probe_name]["mean_correlation"])

    # Colors for wrong task
    wrong_colors = []
    for name in wrong_probe_names:
        if "mean_pooled" in name:
            wrong_colors.append("lightblue")
        else:
            wrong_colors.append("orange")

    # ============ ROW 2: WRONG TASK ============

    # Wrong task MSE (bottom left)
    bars3 = ax3.bar(
        range(len(wrong_probe_names)), wrong_mse_values, color=wrong_colors, alpha=0.7, edgecolor="darkblue"
    )
    ax3.set_title("MSE - wrong task", fontsize=14, fontweight="bold")
    ax3.set_xticks(range(len(wrong_probe_names)))
    ax3.set_xticklabels(wrong_probe_names, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 0.14)  # Increase upper limit to give space for legend

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars3, wrong_mse_values)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(wrong_mse_values) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Wrong task Mean Correlation (bottom right)
    bars4 = ax4.bar(
        range(len(wrong_probe_names)), wrong_mean_corrs, color=wrong_colors, alpha=0.7, edgecolor="darkblue"
    )
    ax4.set_title("Per engine correlation stats - wrong task", fontsize=14, fontweight="bold")
    ax4.set_xticks(range(len(wrong_probe_names)))
    ax4.set_xticklabels(wrong_probe_names, rotation=45, ha="right")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.7, 0.92)

    # Add value labels
    for i, (bar, value) in enumerate(zip(bars4, wrong_mean_corrs)):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Add legend for pooling methods
    from matplotlib.patches import Patch

    legend_elements_correct = [
        Patch(facecolor="steelblue", alpha=0.7, label="Mean Pooled"),
        Patch(facecolor="darkorange", alpha=0.7, label="Last Vector"),
    ]
    legend_elements_wrong = [
        Patch(facecolor="lightblue", alpha=0.7, label="Mean Pooled"),
        Patch(facecolor="orange", alpha=0.7, label="Last Vector"),
    ]

    ax1.legend(handles=legend_elements_correct, loc="upper left", fontsize=10)
    ax3.legend(handles=legend_elements_wrong, loc="upper left", fontsize=10)

    # Main title
    fig.suptitle(
        "Probe analysis - comparing different VLM hidden layers (correct and wrong tasks)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save plot
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    plot_path = os.path.join(output_dir, "layer_comparison_correct_vs_wrong.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}")

    if show_plot:
        plt.show()

    # Print summary
    print("\n📈 Summary:")
    print("-" * 60)

    print("\n🔵 CORRECT TASK:")
    best_mse_idx = np.argmin(correct_mse_values)
    best_corr_idx = np.argmax(correct_mean_corrs)
    print(f"  🏆 Lowest MSE: {correct_probe_names[best_mse_idx]} ({correct_mse_values[best_mse_idx]:.6f})")
    print(f"  🏆 Highest Correlation: {correct_probe_names[best_corr_idx]} ({correct_mean_corrs[best_corr_idx]:.4f})")

    print("\n🔶 WRONG TASK:")
    best_mse_idx = np.argmin(wrong_mse_values)
    best_corr_idx = np.argmax(wrong_mean_corrs)
    print(f"  🏆 Lowest MSE: {wrong_probe_names[best_mse_idx]} ({wrong_mse_values[best_mse_idx]:.6f})")
    print(f"  🏆 Highest Correlation: {wrong_probe_names[best_corr_idx]} ({wrong_mean_corrs[best_corr_idx]:.4f})")

    print("\n" + "=" * 60)


def main():
    """Main function to create layer comparison visualization."""
    print("🔍 VLM Hidden Layer Comparison - Correct vs Wrong Task")
    print("=" * 60)

    visualize_layer_comparison(show_plot=True)

    print("🎉 Visualization completed!")


if __name__ == "__main__":
    main()
