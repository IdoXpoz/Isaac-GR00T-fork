#!/bin/bash
#SBATCH --job-name=compare_feature_types
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/compare_feature_types_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/compare_feature_types_%j.err

echo "Starting feature type comparison at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to the separated_embeddings directory
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/probe/separated_embeddings

# Load conda and activate environment
source /home/morg/students/idoavnir/miniconda/etc/profile.d/conda.sh
conda activate gr00t

# Check if required packages are available
echo "🔍 Checking required packages..."
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} is available')" || {
    echo "❌ NumPy not available"
    exit 1
}

python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__} is available')" || {
    echo "❌ Matplotlib not available"
    exit 1
}

python -c "import pickle; print('✅ Pickle is available')" || {
    echo "❌ Pickle not available"
    exit 1
}

# Get action step from command line argument (default to 0)
ACTION_STEP=${1:-0}
echo "📊 Action step: $ACTION_STEP"

# Check if evaluation metrics exist
echo "🔍 Checking for evaluation metrics..."
METRICS_DIR="/home/morg/students/idoavnir/Isaac-GR00T-fork/probe/separated_embeddings"
if [[ -d "$METRICS_DIR" ]]; then
    echo "✅ Metrics directory found: $METRICS_DIR"
    echo "Contents:"
    ls -la "$METRICS_DIR" | head -20
else
    echo "❌ Metrics directory not found: $METRICS_DIR"
    echo "Please run probe evaluation first."
    exit 1
fi

# Run the comparison script
echo "🚀 Starting feature type comparison for action step $ACTION_STEP..."
python compare_feature_types.py $ACTION_STEP

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Feature type comparison completed successfully at $(date)"
else
    echo "❌ Feature type comparison failed with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
fi

# Show output location
OUTPUT_DIR="$METRICS_DIR/comparisons"
echo "📁 Output saved to: $OUTPUT_DIR"
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Latest comparison plots:"
    ls -lth "$OUTPUT_DIR"/feature_type_comparison_*.png | head -5
fi

echo "🎉 Job completed at $(date)"

