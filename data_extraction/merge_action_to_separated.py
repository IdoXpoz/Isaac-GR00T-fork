#!/usr/bin/env python3
"""
Merge action_right_arm_layer_12 from action_different_vlm_layers_data 
into separated_embeddings_data merged_batches.parquet file.
"""

import pandas as pd
from pathlib import Path


def merge_action_to_separated():
    """
    Merge action_right_arm_layer_12 from action data into separated embeddings data.
    """
    # Define file paths
    base_dir = Path("/home/morg/students/idoavnir/Isaac-GR00T-fork")
    action_file = base_dir / "action_different_vlm_layers_data" / "batches_parquet" / "merged_batches.parquet"
    separated_file = base_dir / "separated_embeddings_data" / "batches_parquet" / "merged_batches.parquet"
    
    print(f"{'='*80}")
    print(f"🔄 MERGING ACTION DATA TO SEPARATED EMBEDDINGS")
    print(f"{'='*80}")
    print(f"Action data file: {action_file}")
    print(f"Separated data file: {separated_file}")
    print()
    
    # Check if files exist
    if not action_file.exists():
        raise FileNotFoundError(f"Action data file not found: {action_file}")
    if not separated_file.exists():
        raise FileNotFoundError(f"Separated data file not found: {separated_file}")
    
    # Load both parquet files
    print("📂 Loading action data parquet...")
    action_df = pd.read_parquet(action_file)
    print(f"   • Loaded {len(action_df)} rows, {len(action_df.columns)} columns")
    
    print("📂 Loading separated embeddings parquet...")
    separated_df = pd.read_parquet(separated_file)
    print(f"   • Loaded {len(separated_df)} rows, {len(separated_df.columns)} columns")
    print()
    
    # Verify the column exists
    if 'action_right_arm_layer_12' not in action_df.columns:
        raise ValueError(f"Column 'action_right_arm_layer_12' not found in action data")
    
    # Verify row counts match
    if len(action_df) != len(separated_df):
        print(f"⚠️  WARNING: Row count mismatch!")
        print(f"   Action data: {len(action_df)} rows")
        print(f"   Separated data: {len(separated_df)} rows")
    
    # Extract the action column and rename it
    print("🔧 Extracting action_right_arm_layer_12 column...")
    separated_df['action_right_arm'] = action_df['action_right_arm_layer_12'].values
    
    print(f"✅ Added 'action_right_arm' column to separated data")
    print()
    
    # Display updated info
    print("📊 Updated Separated Data Info:")
    print(f"   • Rows: {len(separated_df)}")
    print(f"   • Columns: {len(separated_df.columns)}")
    print(f"   • Column names: {list(separated_df.columns)}")
    print()
    
    # Save the updated parquet file
    print(f"💾 Saving updated parquet to: {separated_file}")
    separated_df.to_parquet(separated_file, index=False)
    
    # Verify file size
    file_size_mb = separated_file.stat().st_size / (1024 * 1024)
    print(f"✅ File saved successfully!")
    print(f"   • New file size: {file_size_mb:.1f} MB")
    print(f"{'='*80}")
    print("🎉 Merge completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    merge_action_to_separated()

