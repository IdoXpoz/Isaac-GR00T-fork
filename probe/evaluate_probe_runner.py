#!/usr/bin/env python3
"""
Runner script for comparing all probes for action_different_vlm_layers data.
"""

import os
import sys

# Add parent directory to path to import evaluate_probe module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.evaluate_probe import compare_all_probes_for_action_step


def run():
    """Run probe comparison for action_different_vlm_layers data."""
    print("🏁 Starting probe comparison for action_different_vlm_layers data...")
    print("=" * 60)

    # Get action step from command line argument if provided
    action_step = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print(f"📊 Action step: {action_step}")
    print(f"📁 Data source: action_different_vlm_layers_data")
    print("=" * 60)

    # Run comparison with specified action step
    results = compare_all_probes_for_action_step(
        action_step=action_step,
        show_plot=True,
    )

    if results:
        print("\n" + "=" * 60)
        print("📊 SUMMARY OF ALL PROBES")
        print("=" * 60)
        for probe_name, metrics in results.items():
            print(f"\n{probe_name}:")
            print(f"  MSE:              {metrics['mse']:.6f}")
            print(f"  Mean Correlation: {metrics['mean_correlation']:.4f}")
            print(f"  Max Correlation:  {metrics['max_correlation']:.4f}")
            print(f"  Min Correlation:  {metrics['min_correlation']:.4f}")

    print("\n" + "=" * 60)
    print("🎉 Comparison completed!")
    print("=" * 60)


if __name__ == "__main__":
    run()
