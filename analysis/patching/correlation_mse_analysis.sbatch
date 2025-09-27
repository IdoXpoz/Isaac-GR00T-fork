#!/bin/bash
#SBATCH --job-name=correlation_mse_analysis
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/correlation_mse_analysis_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/correlation_mse_analysis_%j.err

echo "Starting correlation and MSE analysis at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to the analysis directory
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/analysis/patching

# Load conda and activate environment
source /home/morg/students/idoavnir/miniconda/etc/profile.d/conda.sh
conda activate gr00t

# Install seaborn if not already available
echo "📦 Installing seaborn..."
conda install seaborn -y

# Install scikit-learn if not already available
echo "📦 Installing scikit-learn..."
conda install scikit-learn -y

# Install scipy if not already available
echo "📦 Installing scipy..."
conda install scipy -y

# Check if required packages are available
python -c "import pandas; print(f'✅ Pandas {pandas.__version__} is available')" || {
    echo "❌ Pandas not available"
    exit 1
}

python -c "import numpy; print(f'✅ NumPy {numpy.__version__} is available')" || {
    echo "❌ NumPy not available"
    exit 1
}

python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__} is available')" || {
    echo "❌ Matplotlib not available"
    exit 1
}

python -c "import seaborn; print(f'✅ Seaborn {seaborn.__version__} is available')" || {
    echo "❌ Seaborn not available"
    exit 1
}

python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__} is available')" || {
    echo "❌ Scikit-learn not available"
    exit 1
}

python -c "import scipy; print(f'✅ SciPy {scipy.__version__} is available')" || {
    echo "❌ SciPy not available"
    exit 1
}

# Check data availability
echo "🔍 Checking data availability..."
DATA_FILE="/home/morg/students/idoavnir/Isaac-GR00T-fork/action_different_vlm_layers_data/batches_parquet/merged_batches.parquet"
if [[ -f "$DATA_FILE" ]]; then
    echo "✅ Data file found: $DATA_FILE"
    ls -lh "$DATA_FILE"
else
    echo "❌ Data file not found: $DATA_FILE"
    echo "Available files in batches_parquet directory:"
    ls -la /home/morg/students/idoavnir/Isaac-GR00T-fork/action_different_vlm_layers_data/batches_parquet/
    exit 1
fi

# Run the analysis script
echo "🚀 Starting correlation and MSE analysis..."
python correlation_and_mse_check.py

echo "✅ Correlation and MSE analysis job completed at $(date)"
