import os
import torch
from data_extraction.utils.batches import extract_batches, merge_batches
from data_extraction.utils.extraction_functions import extract_single_step_data
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP


def main():
    MODEL_PATH = "nvidia/GR00T-N1.5-3B"
    DATASET_ROOT = "/home/morg/students/idoavnir/Isaac-GR00T-fork/gr00t_dataset"
    OUTPUT_DIR = "/home/morg/students/idoavnir/Isaac-GR00T-fork/vlm_by_layers_raise_hands"
    EMBODIMENT_TAG = "gr1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")

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

        extract_batches(policy, OUTPUT_DIR)

        print("✅ Batches extracted successfully!")
        print("merging batches...")

        merge_batches(OUTPUT_DIR)

        print("✅ Batches merged successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
