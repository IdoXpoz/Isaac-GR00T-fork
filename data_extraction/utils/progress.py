import json
import glob
from datetime import datetime
import os
import torch
import pandas as pd


def load_progress(output_dir: str):
    """Load extraction progress from file"""
    PROGRESS_FILE = os.path.join(output_dir, "extraction_progress_parquet.json")
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_batches": [], "total_extracted": 0, "last_batch_id": 0, "start_time": datetime.now().isoformat()}


def save_progress(progress, output_dir: str):
    """Save extraction progress to file"""
    PROGRESS_FILE = os.path.join(output_dir, "extraction_progress_parquet.json")
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
