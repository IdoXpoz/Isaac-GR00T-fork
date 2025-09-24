#!/bin/bash
#SBATCH --job-name=extract_action_layers
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/extract_action_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/extract_action_%j.err

echo "Starting action extraction using different VLM layers at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU info:"
nvidia-smi

# Navigate to the script directory
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/data_extraction/action_different_vlm_layers

# Set Hugging Face cache to current working directory to avoid quota issues
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

python -c "import gr00t; print('✅ GR00T is available')" || {
    echo "❌ GR00T not available"
    exit 1
}

# Check CUDA availability
python -c "import torch; print(f'🔍 CUDA available: {torch.cuda.is_available()}')"

# Run the extraction script
echo "Starting action extraction using different VLM layers..."
python action_different_vlm_layers_extractor.py

echo "Action extraction job completed at $(date)"
