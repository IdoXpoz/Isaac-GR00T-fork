# GR00T Probe Analysis

This directory contains scripts for training and evaluating probes on the GR00T robotic model. The probe aims to understand whether the VLM's intermediate representations contain information predictive of the final action outputs.

## Overview

The probe analysis consists of:
1. **Data Extraction**: Extract VLM backbone features and action targets (done in `../getting_started/extract_probe_training_data.ipynb`)
2. **Probe Training**: Train a neural network to predict action tokens from VLM features
3. **Evaluation**: Analyze the trained probe's performance

## Files

- `train_probe.py`: Main training script for the probe model
- `evaluate_probe.py`: Evaluation and analysis script  
- `README.md`: This documentation file

## Quick Start

### 1. Extract Training Data

First, run the data extraction notebook to create the probe training data:

```bash
# Run the notebook: getting_started/extract_probe_training_data.ipynb
# This will create processed data: probe_training_data_150k_processed.parquet
```

### 2. Train the Probe

```bash
cd probe
python train_probe.py
```

This will:
- Load the processed data from `probe_training_data_150k_processed.parquet`
- Use pre-computed mean-pooled or last vector features [2048]
- Split data into 99% train, 1% test
- Train a linear regression probe (no hidden layers)
- Save the best model to `best_probe_model.pth`
- Save training history to `training_history.pkl`

### 3. Evaluate the Probe

```bash
python evaluate_probe.py
```

This will:
- Load the trained model
- Evaluate on test data
- Generate performance plots
- Save detailed metrics

## Model Architecture

The probe uses a simple linear regression architecture:

```
Input: Pre-processed VLM Backbone Features [2048] 
       (mean pooled or last vector)
   ↓
Linear Regression: [2048] → [action_dim]
   ↓
Output: Action Predictions [action_dim]
```

**Feature Types Available:**
- `mean_pooled`: Mean pooling across sequence dimension
- `last_vector`: Last token from sequence

## Data Format

The probe expects processed data in parquet format with the following columns:

```
- backbone_features_mean_pooled: Pre-computed mean-pooled features [2048]
- backbone_features_last_vector: Pre-computed last vector features [2048]
- action_right_arm_first: Action target values
- Additional metadata columns (sample_id, task_name, etc.)
```

## Configuration

Key hyperparameters in `train_probe.py`:

```python
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FEATURE_TYPE = "mean_pooled"  # Options: "mean_pooled" or "last_vector"
TRAIN_RATIO = 0.99  # 99% train, 1% test
DATA_PATH = "probe_training_data_150k_processed.parquet"  # Processed parquet format
```

**To switch between feature types**, simply change `FEATURE_TYPE` to:
- `"mean_pooled"`: Use mean pooling across sequence dimension
- `"last_vector"`: Use the last vector from sequence

## Output Files

After training and evaluation:

```
probe/
├── best_probe_model.pth           # Trained model weights
├── training_history.pkl           # Training loss curves
├── evaluation_metrics.pkl         # Detailed evaluation metrics
├── training_curves.png           # Training/validation loss plots
└── predictions_vs_targets.png    # Prediction scatter plots
```

## Understanding Results

### Good Probe Performance Indicators:
- **High Correlation (>0.6)**: Strong linear relationship between VLM features and actions
- **Low MSE/RMSE**: Accurate numerical predictions
- **Smooth Training Curves**: Model converged properly

### What the Results Mean:
- **High Performance**: VLM features contain rich information about future actions
- **Low Performance**: VLM features may not be directly predictive, or more complex modeling needed
- **Dimension-wise Analysis**: Some action dimensions may be more predictable than others

## Troubleshooting

### Common Issues:

1. **Data Not Found Error**:
   ```
   ❌ Data file not found: probe_training_data_150k.parquet
   ```
   Solution: Run the data extraction notebook first

2. **CUDA Out of Memory**:
   ```
   torch.cuda.OutOfMemoryError
   ```
   Solution: Reduce `BATCH_SIZE` in the training script

3. **Poor Performance**:
   - Try different feature types (`mean_pooled` vs `last_vector`)
   - Since it's linear regression, consider adding regularization
   - Experiment with different learning rates

### Memory Requirements:
- GPU: ~2-4GB VRAM (depending on batch size)
- RAM: ~8-16GB (depending on dataset size)

## Extending the Analysis

### Custom Probe Architectures:
Modify the `ActionProbe` class in `train_probe.py` to experiment with:
- Add regularization techniques (L1/L2, dropout)
- Multi-task prediction (predict multiple action types)
- Non-linear transformations before linear layer
- Ensemble methods with different feature types

### Multiple Target Analysis:
Currently predicts only `action_right_arm_first`. You can extend to:
- Multiple action dimensions
- Different action types (left arm, waist, etc.)
- Temporal sequences of actions

### Advanced Analysis:
- Feature importance analysis
- Gradient-based interpretability
- Layer-wise probe comparison
- Cross-dataset generalization

## References

This probe analysis is inspired by research on:
- Probing neural network representations
- Vision-language model interpretability  
- Robotics action prediction

For questions or issues, please refer to the main GR00T repository documentation.