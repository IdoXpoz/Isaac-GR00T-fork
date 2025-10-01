#!/bin/bash
#SBATCH --job-name=evaluate_probe
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/evaluate_probe_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/evaluate_probe_%j.err

echo "Starting probe evaluation at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi

# Navigate to the probe directory
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/probe

# Set Hugging Face cache to avoid quota issues
export HF_HOME=/home/morg/students/idoavnir/Isaac-GR00T-fork/.cache/huggingface
export HF_DATASETS_CACHE=/home/morg/students/idoavnir/Isaac-GR00T-fork/.cache/huggingface/datasets
export HF_MODULES_CACHE=/home/morg/students/idoavnir/Isaac-GR00T-fork/.cache/huggingface/modules
export TRANSFORMERS_CACHE=/home/morg/students/idoavnir/Isaac-GR00T-fork/.cache/huggingface/transformers

# Create cache directory
mkdir -p /home/morg/students/idoavnir/Isaac-GR00T-fork/.cache/huggingface

echo "🔍 HF_HOME set to: $HF_HOME"

# Load conda and activate environment
source /home/morg/students/idoavnir/miniconda/etc/profile.d/conda.sh
conda activate gr00t

# Check if required packages are available
python -c "import torch; print(f'✅ PyTorch {torch.__version__} is available')" || {
    echo "❌ PyTorch not available"
    exit 1
}

python -c "import numpy; print(f'✅ NumPy {numpy.__version__} is available')" || {
    echo "❌ NumPy not available"
    exit 1
}

python -c "import pandas; print(f'✅ Pandas {pandas.__version__} is available')" || {
    echo "❌ Pandas not available"
    exit 1
}

python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__} is available')" || {
    echo "❌ Matplotlib not available"
    exit 1
}

# Check CUDA availability
python -c "import torch; print(f'🔍 CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'🔍 GPU count: {torch.cuda.device_count()}')"

# Run the evaluation script
echo "Starting probe evaluation..."
python evaluate_probe_runner.py

echo "Probe evaluation job completed at $(date)"

