#!/usr/bin/env python3
"""
GR00T Probe Evaluation Script

This script evaluates a trained probe model and provides detailed analysis
of its performance in predicting robot action tokens from VLM features.

Author: Generated for GR00T probe analysis
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from train_probe import ActionProbe, ProbeDataset, load_probe_data, split_data


def load_trained_model(model_path: str, input_dim: int, output_dim: int) -> ActionProbe:
    """Load a trained probe model."""
    model = ActionProbe(input_dim=input_dim, output_dim=output_dim)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def detailed_evaluation(model: ActionProbe, dataloader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Perform detailed evaluation of the probe model."""
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    losses = []

    criterion = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            losses.append(loss.cpu().numpy())

    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    losses = np.concatenate(losses, axis=0)

    # Calculate metrics
    mse = np.mean(losses)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # Per-dimension metrics if multi-dimensional
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        per_dim_mse = np.mean(losses, axis=0)
        per_dim_rmse = np.sqrt(per_dim_mse)
        per_dim_mae = np.mean(np.abs(predictions - targets), axis=0)
    else:
        per_dim_mse = [mse]
        per_dim_rmse = [rmse]
        per_dim_mae = [mae]

    # Correlation coefficient
    if len(targets.shape) > 1:
        correlations = []
        for i in range(targets.shape[1]):
            corr = np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
    else:
        correlations = [np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]]

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "per_dim_mse": per_dim_mse,
        "per_dim_rmse": per_dim_rmse,
        "per_dim_mae": per_dim_mae,
        "correlations": correlations,
        "predictions": predictions,
        "targets": targets,
    }


def plot_training_history(history_path: str, save_path: str = "probe/training_curves.png"):
    """Plot training curves."""
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss (Zoomed)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved to: {save_path}")


def plot_predictions_vs_targets(
    predictions: np.ndarray, targets: np.ndarray, save_path: str = "probe/predictions_vs_targets.png"
):
    """Plot predictions vs targets."""
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # Multi-dimensional output
        n_dims = min(targets.shape[1], 4)  # Plot max 4 dimensions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i in range(n_dims):
            ax = axes[i]
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

            ax.set_xlabel(f"True Values (Dim {i})")
            ax.set_ylabel(f"Predictions (Dim {i})")
            ax.set_title(f"Dimension {i}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for i in range(n_dims, 4):
            axes[i].set_visible(False)

    else:
        # Single dimensional output
        plt.figure(figsize=(8, 6))
        plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("Predictions vs True Values")
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Prediction plots saved to: {save_path}")


def print_evaluation_summary(metrics: Dict[str, float]):
    """Print a detailed evaluation summary."""
    print("\n" + "=" * 60)
    print("PROBE EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n📊 Overall Performance:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")

    if len(metrics["per_dim_mse"]) > 1:
        print(f"\n📏 Per-Dimension Performance:")
        for i, (mse, rmse, mae, corr) in enumerate(
            zip(metrics["per_dim_mse"], metrics["per_dim_rmse"], metrics["per_dim_mae"], metrics["correlations"])
        ):
            print(f"  Dim {i}: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, Corr={corr:.4f}")

    print(f"\n🔗 Correlation Analysis:")
    avg_corr = np.mean(metrics["correlations"])
    print(f"  Average Correlation: {avg_corr:.4f}")

    if avg_corr > 0.8:
        quality = "Excellent 🌟"
    elif avg_corr > 0.6:
        quality = "Good ✅"
    elif avg_corr > 0.4:
        quality = "Moderate ⚠️"
    else:
        quality = "Poor ❌"

    print(f"  Quality Assessment: {quality}")

    print("\n" + "=" * 60)


def main(feature_type: str = "mean_pooled", data_path: str = None, model_path: str = None):
    """Main evaluation function.
    
    Args:
        feature_type: Type of features to use - should match training configuration
        data_path: Path to the processed data file (optional)
        model_path: Path to the trained model file (optional)
    """
    # Configuration
    MODEL_PATH = model_path or "probe/best_probe_model.pth"
    DATA_PATH = data_path or "probe_training_data_150k_processed.parquet"  # Use processed data
    FEATURE_TYPE = feature_type
    HISTORY_PATH = "probe/training_history.pkl"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    print(f"Feature type: {FEATURE_TYPE}")

    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the probe first using train_probe.py")
        return

    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Please make sure you've run the data extraction and processing notebook first.")
        return

    # Load data with specified feature type
    backbone_features, action_targets = load_probe_data(DATA_PATH, feature_type=FEATURE_TYPE)

    # Split data (same split as training)
    _, _, test_features, test_targets = split_data(backbone_features, action_targets, train_ratio=0.99)

    # Create test dataset
    test_dataset = ProbeDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get dimensions
    sample_features, sample_target = test_dataset[0]
    input_dim = sample_features.shape[-1]  # Should be 2048
    output_dim = sample_target.shape[-1] if len(sample_target.shape) > 0 else 1

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Test samples: {len(test_dataset)}")

    # Load trained model (linear regression)
    model = load_trained_model(MODEL_PATH, input_dim, output_dim)
    print(f"✅ Loaded trained linear regression model from {MODEL_PATH}")

    # Evaluate model
    print("\n🔍 Evaluating model...")
    metrics = detailed_evaluation(model, test_loader, device=DEVICE)

    # Print summary
    print_evaluation_summary(metrics)

    # Plot training history if available
    if os.path.exists(HISTORY_PATH):
        print("\n📈 Plotting training curves...")
        plot_training_history(HISTORY_PATH)

    # Plot predictions vs targets
    print("\n📊 Plotting predictions vs targets...")
    plot_predictions_vs_targets(metrics["predictions"], metrics["targets"])

    # Save detailed metrics
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ["predictions", "targets"]}
    with open("probe/evaluation_metrics.pkl", "wb") as f:
        pickle.dump(metrics_to_save, f)

    print(f"\n💾 Evaluation metrics saved to: probe/evaluation_metrics.pkl")
    print(f"Feature type used: {FEATURE_TYPE}")
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
