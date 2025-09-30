#!/bin/bash
#SBATCH --job-name=inspect_batch_0001
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/batch_inspection/inspect_batch_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/batch_inspection/inspect_batch_%j.err

echo "Starting parquet batch inspection at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to the batch_inspection directory where the script is located
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/batch_inspection

# Load conda and activate environment
source /home/morg/students/idoavnir/miniconda/etc/profile.d/conda.sh
conda activate gr00t

# Check if required packages are available
python -c "import pandas; print(f'✅ Pandas {pandas.__version__} is available')" || {
    echo "❌ Pandas not available"
    exit 1
}

python -c "import os; print('✅ OS module is available')" || {
    echo "❌ OS module not available"
    exit 1
}

# Run the inspection script
echo "Starting parquet file inspection..."
python inspect_batch.py

echo "Parquet inspection job completed at $(date)"
