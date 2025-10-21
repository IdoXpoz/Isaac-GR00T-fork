#!/bin/bash
#SBATCH --job-name=merge_action_separated
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/merge_action_separated_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/merge_action_separated_%j.err

echo "Starting action data merge to separated embeddings at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to the data_extraction directory where the script is located
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction

# Load conda and activate environment
source /home/morg/students/idoavnir/miniconda/etc/profile.d/conda.sh
conda activate gr00t

# Check if required packages are available
python -c "import pandas; print(f'✅ Pandas {pandas.__version__} is available')" || {
    echo "❌ Pandas not available"
    exit 1
}

python -c "from pathlib import Path; print('✅ pathlib is available')" || {
    echo "❌ pathlib not available"
    exit 1
}

# Run the merge script
echo "Starting merge operation on separated embeddings merged_batches.parquet..."
python merge_action_to_separated.py

echo "Merge job completed at $(date)"

