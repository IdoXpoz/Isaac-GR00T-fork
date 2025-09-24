import os
import torch

from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from data_extraction.utils.batches import extract_batches, merge_batches

MODEL_PATH = "nvidia/GR00T-N1.5-3B"
DATASET_ROOT = "/home/morg/students/idoavnir/Isaac-GR00T-fork/gr00t_dataset"
OUTPUT_DIR = "/home/morg/students/idoavnir/Isaac-GR00T-fork/vlm_by_layers_raise_hands"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBODIMENT_TAG = "gr1"
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_policy():
    try:
        data_config = DATA_CONFIG_MAP["fourier_gr1_arms_waist"]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=modality_config,
            modality_transform=modality_transform,
            device=device,
        )
        print("✅ Policy loaded successfully!")
        return policy
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        raise


def run():
    policy = create_policy()
    extract_batches(policy, OUTPUT_DIR)
    print("\n🎉 Batch extraction completed!")

    processed_file, total_samples = merge_batches(OUTPUT_DIR)
    if processed_file:
        print(f"\n🎉 Processed parquet file ready!")
        print(f"📁 Location: {processed_file}")
        print(f"📊 Total samples: {total_samples:,}")
    else:
        print("❌ No processed parquet file found")


if __name__ == "__main__":
    run()
