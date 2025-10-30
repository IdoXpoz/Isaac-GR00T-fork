#!/bin/bash
#SBATCH --job-name=visualize_layers
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=/home/morg/students/idoavnir/Isaac-GR00T-fork/visualize_layers_%j.out
#SBATCH --error=/home/morg/students/idoavnir/Isaac-GR00T-fork/visualize_layers_%j.err

echo "Starting layer comparison visualization at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Navigate to the hidden_layers directory
cd /home/morg/students/idoavnir/Isaac-GR00T-fork/probe/hidden_layers

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

# Check if data file exists
if [[ -f "probe_comparison_data.py" ]]; then
    echo "✅ Data file found: probe_comparison_data.py"
else
    echo "❌ Data file not found: probe_comparison_data.py"
    exit 1
fi

# Run the visualization script
echo "🚀 Creating layer comparison visualization..."
python visualize_layer_comparison.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Visualization completed successfully at $(date)"
else
    echo "❌ Visualization failed with exit code $EXIT_CODE at $(date)"
    exit $EXIT_CODE
fi

# Show output
echo "📁 Output:"
if [[ -f "layer_comparison_correct_vs_wrong.png" ]]; then
    ls -lh layer_comparison_correct_vs_wrong.png
fi

echo "🎉 Job completed at $(date)"

