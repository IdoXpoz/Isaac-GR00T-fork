#!/usr/bin/env python3
"""
Runner script for training probes on GR00T fused embeddings.
"""

import os
import sys

# Add parent directory to path to import train_probe module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe.train_probe import train_single_probe


def run():
    """Run probe training."""
    print("🏁 Starting probe training...")
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

    # Run training with specified parameters
    train_single_probe(
        data_path=data_path,
        feature_col_name="vision_mean_pooled",  # Can be modified as needed
        batch_size=32,
        num_epochs=100,
        action_step=0,
    )

    print("=" * 60)
    print("🎉 Training completed!")


if __name__ == "__main__":
    run()
