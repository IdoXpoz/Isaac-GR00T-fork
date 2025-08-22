#!/usr/bin/env python3
"""
Download GR1 Arms Waist Dataset

This script downloads specific tasks from the nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
dataset for the GR1 arms and waist embodiment to the home folder.
"""

import os
from huggingface_hub import snapshot_download


def main():
    # Define the dataset repository
    REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
    LOCAL_DIR = "./gr00t_dataset"  # Current repository

    # Define the tasks to download
    TASKS = [
        # "gr1_arms_waist.CanToDrawer",
        # "gr1_arms_waist.CupToDrawer",
        # "gr1_arms_waist.CuttingBoardToBasket",
        # "gr1_arms_waist.CuttingBoardToCardboardBox",
        # "gr1_arms_waist.CuttingBoardToPan",
        # "gr1_arms_waist.CuttingBoardToPot",
        # "gr1_arms_waist.PlaceBottleToCabinet",
        # "gr1_arms_waist.PlaceMilkToMicrowave",
        # "gr1_arms_waist.PlacematToBowl",
        # "gr1_arms_waist.PotatoToMicrowave",
        "gr1_arms_waist.TrayToPot",
        # "gr1_arms_waist.TrayToTieredShelf",
    ]

    # Create include patterns for all tasks
    include_patterns = []
    for task in TASKS:
        include_patterns.extend([f"{task}/meta/**", f"{task}/data/chunk-000/**", f"{task}/videos/chunk-000/**"])

    print(f"Downloading {len(TASKS)} tasks to: {LOCAL_DIR}")
    print("This may take several minutes depending on the dataset size and your connection speed.")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            allow_patterns=include_patterns,
            resume_download=True,  # Resume if interrupted
        )
        print(f"✅ Download completed successfully!")
        print(f"Dataset saved to: {LOCAL_DIR}")

    except Exception as e:
        print(f"❌ Download failed with error: {str(e)}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()
