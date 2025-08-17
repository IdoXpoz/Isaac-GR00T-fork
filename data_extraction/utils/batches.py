import os
import json
import glob
import pandas as pd
import tqdm
from datetime import datetime
from data_extraction.utils.progress import load_progress, save_progress
from data_extraction.vlm_layers.vlm_layers_extractor import extract_single_step_data
from data_extraction.utils.modality_config import modality_config
from data_extraction.utils.dataset import LeRobotSingleDataset
from data_extraction.utils.policy import Gr00tPolicy

TARGET_TOTAL_SAMPLES = 60000
BATCH_SIZE = 1000
TASK_NAME = "gr1_arms_waist.TrayToPot"
DATASET_ROOT = "/content/drive/MyDrive/gr1_arms_waist"
EMBODIMENT_TAG = "gr1"
MODEL_PATH = "nvidia/GR00T-N1.5-3B"
MERGED_BATCHES_FILE_NAME = "merged_batches.parquet"


def get_batch_filename(batch_id, output_dir: str):
    """Get standardized batch filename"""
    batch_output_dir = os.path.join(output_dir, "batches_parquet")
    return os.path.join(batch_output_dir, f"batch_{batch_id:04d}.parquet")


def save_batch_data(batch_data, batch_id):
    """Save a single batch to parquet file"""

    # Prepare data for DataFrame
    rows = []

    for _, data in enumerate(batch_data):
        rows.append(data)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save as parquet with compression
    batch_file = get_batch_filename(batch_id)
    df.to_parquet(batch_file, compression="snappy", index=False)

    # Save batch metadata separately
    batch_metadata = {
        "batch_id": batch_id,
        "batch_size": len(batch_data),
        "extraction_time": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "embodiment_tag": EMBODIMENT_TAG,
        "file_format": "parquet",
        "compression": "snappy",
    }

    metadata_file = batch_file.replace(".parquet", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(batch_metadata, f, indent=2)

    return batch_file, len(batch_data)


def extract_batches(
    policy: Gr00tPolicy,
    output_dir: str,
):
    """
    Main function to extract data in batches with resume capability
    """

    # Load existing progress
    progress = load_progress(output_dir)
    print(f"📊 Current progress: {progress['total_extracted']:,} samples extracted")

    if progress["total_extracted"] >= TARGET_TOTAL_SAMPLES:
        print(f"✅ Target already reached! {progress['total_extracted']:,} >= {TARGET_TOTAL_SAMPLES:,}")
        return

    # Load dataset
    task_path = os.path.join(DATASET_ROOT, TASK_NAME)

    print(f"🔄 Loading dataset: {TASK_NAME}")
    dataset = LeRobotSingleDataset(
        dataset_path=task_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=EMBODIMENT_TAG,
    )

    print(f"📊 Dataset size: {len(dataset):,} samples")

    # Calculate remaining work
    remaining_samples = TARGET_TOTAL_SAMPLES - progress["total_extracted"]
    start_index = progress["total_extracted"]

    print(f"🎯 Need to extract {remaining_samples:,} more samples")
    print(f"▶️  Starting from index: {start_index:,}")

    # Process in batches
    current_batch_data = []
    batch_id = progress["last_batch_id"] + 1
    samples_processed = 0

    try:
        for i in tqdm(
            range(start_index, min(start_index + remaining_samples, len(dataset))), desc="Extracting batches"
        ):
            try:
                # Get sample data
                step_data = dataset[i]

                # Create dataset info
                dataset_info = {
                    "task_name": TASK_NAME,
                    "sample_index": i,
                    "total_samples": len(dataset),
                    "global_index": progress["total_extracted"] + samples_processed,
                }

                # Extract data
                data_dict = extract_single_step_data(policy, step_data, dataset_info)
                current_batch_data.append(data_dict)
                samples_processed += 1

                if len(current_batch_data) >= BATCH_SIZE:
                    _, batch_size_actual = save_batch_data(current_batch_data, batch_id)

                    # Update progress
                    progress["completed_batches"].append(batch_id)
                    progress["total_extracted"] += batch_size_actual
                    progress["last_batch_id"] = batch_id
                    save_progress(progress)

                    print(
                        f"✅ Saved batch {batch_id:04d}: {batch_size_actual:,} samples → {progress['total_extracted']:,} total"
                    )

                    # Clear batch data and increment batch_id
                    current_batch_data = []
                    batch_id += 1

                    # Check if we've reached our target
                    if progress["total_extracted"] >= TARGET_TOTAL_SAMPLES:
                        print(f"🎉 Target reached! {progress['total_extracted']:,} samples extracted")
                        break

            except Exception as e:
                print(f"⚠️  Failed to process sample {i}: {e}")
                continue

        # Save any remaining data in the last batch
        if current_batch_data:
            _, batch_size_actual = save_batch_data(current_batch_data, batch_id)
            progress["completed_batches"].append(batch_id)
            progress["total_extracted"] += batch_size_actual
            progress["last_batch_id"] = batch_id
            save_progress(progress)
            print(
                f"✅ Saved final batch {batch_id:04d}: {batch_size_actual:,} samples → {progress['total_extracted']:,} total"
            )

    except Exception as e:
        print(f"❌ Error during extraction: {e}")

    # Final summary
    final_progress = load_progress()
    print(f"\n📊 Extraction summary:")
    print(f"   • Total extracted: {final_progress['total_extracted']:,} samples")
    print(f"   • Batches completed: {len(final_progress['completed_batches'])}")
    print(f"   • Progress: {final_progress['total_extracted']/TARGET_TOTAL_SAMPLES*100:.1f}%")

    return final_progress


def merge_batches(output_dir: str):
    """
    Merge batch files into a single file without any data manipulation.
    Simply concatenates all data from batch files.
    """
    batch_output_dir = os.path.join(output_dir, "batches_parquet")

    print(f"Getting files from {batch_output_dir}/batch_*.parquet")

    # Find all batch files
    batch_files = glob.glob(os.path.join(batch_output_dir, "batch_*.parquet"))
    batch_files.sort()  # Sort to process in order

    if not batch_files:
        print("❌ No batch files found to merge!")
        return None

    print(f"🔗 Found {len(batch_files)} parquet batch files for merging")
    print(f"🎯 Creating merged file without any data manipulation")

    # Read and concatenate all batch files
    all_dataframes = []
    total_samples = 0

    for batch_file in batch_files:
        try:
            # Read batch parquet file
            df = pd.read_parquet(batch_file)
            print(f"📁 Processing {os.path.basename(batch_file)}: {len(df)} rows")

            all_dataframes.append(df)
            total_samples += len(df)

        except Exception as e:
            print(f"⚠️  Error processing {batch_file}: {e}")
            continue

    if not all_dataframes:
        print("❌ No valid batch files could be read!")
        return None

    # Concatenate all DataFrames
    print(f"🔄 Concatenating {len(all_dataframes)} batch files with {total_samples:,} total samples...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)

    # Save merged file
    final_output_file = os.path.join(batch_output_dir, MERGED_BATCHES_FILE_NAME)

    print(f"💾 Saving merged parquet file to: {final_output_file}")
    merged_df.to_parquet(final_output_file, compression="snappy", index=False)

    print(f"✅ Merge completed!")
    print(f"   • Total samples: {total_samples:,}")
    print(f"   • Columns: {merged_df.columns}")
    print(f"   • num of columns: {len(merged_df.columns)}")

    return final_output_file, total_samples
