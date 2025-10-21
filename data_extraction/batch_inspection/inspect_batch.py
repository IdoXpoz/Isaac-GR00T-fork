#!/usr/bin/env python3
"""
Script to inspect a specific parquet batch file using the inspect_parquet_file function.
This script inspects action_different_vlm_layers_data/batches_parquet/batch_0001.parquet
"""

import os
import sys

# Add the project root to Python path to enable imports
project_root = "/home/morg/students/idoavnir/Isaac-GR00T-fork"
sys.path.insert(0, project_root)

from data_extraction.utils.parquet import inspect_parquet_file


def main():
    """Main function to inspect the specified parquet file"""
    
    # Define the parquet file path
    parquet_file_path = "/home/morg/students/idoavnir/Isaac-GR00T-fork/separated_embeddings_data/batches_parquet/merged_batches.parquet"
    
    print("=" * 80)
    print("🔍 PARQUET BATCH INSPECTION")
    print("=" * 80)
    print(f"Target file: {parquet_file_path}")
    print(f"Script running from: {os.getcwd()}")
    print("=" * 80)
    
    # Run the inspection
    inspect_parquet_file(parquet_file_path)
    
    print("=" * 80)
    print("✅ Inspection completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
