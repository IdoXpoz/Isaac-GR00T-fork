#!/usr/bin/env python3
"""
Runner script for evaluating trained probes on GR00T fused embeddings.
"""

import os
import sys

# Add parent directory to path to import evaluate_probe module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.evaluate_probe import evaluate_single_probe


def run():
    """Run probe evaluation."""
    print("🏁 Starting probe evaluation...")
    print("=" * 60)

    # Configure paths
    data_path = (
        "/home/morg/students/idoavnir/Isaac-GR00T-fork/separated_embeddings_data/batches_parquet/merged_batches.parquet"
    )

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please make sure the data extraction has been completed.")
        sys.exit(1)

    print(f"✅ Found data file: {data_path}")

    # Run evaluation with specified parameters
    evaluate_single_probe(
        feature_col_name="vision_last_vector",  # Can be modified as needed
        action_step=0,
        data_path=data_path,
    )

    print("=" * 60)
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    run()
