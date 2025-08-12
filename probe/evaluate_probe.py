#!/usr/bin/env python3
"""
GR00T Probe Evaluation Script

This script evaluates a trained probe model and provides detailed analysis
of its performance in predicting robot action tokens from VLM features.

Author: Generated for GR00T probe analysis
"""

import os
import pickle
import csv
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple

from probe.train_probe import ActionProbe, ProbeDataset, load_probe_data


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


def plot_training_history(history_path: str, save_path: str = None):
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


def plot_predictions_vs_targets(predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """Plot predictions vs targets."""
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # Multi-dimensional output
        n_dims = targets.shape[1]
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
    print(f"  correlations per dimension: {metrics['correlations']}")

    print("\n" + "=" * 60)


def _configure_paths(feature_type: str, data_path: str, model_path: str) -> Tuple[str, str, str, str, str]:
    """Return output dir, model path, data path, history path, feature type label."""
    probe_output_dir = f"/content/drive/MyDrive/probes/{feature_type}"
    model_path_final = model_path or os.path.join(probe_output_dir, "best_probe_model.pth")
    data_path_final = (
        data_path or "/content/drive/MyDrive/probe_training_data/probe_training_data_60k_processed.parquet"
    )
    history_path = os.path.join(probe_output_dir, "training_history.pkl")
    return probe_output_dir, model_path_final, data_path_final, history_path, feature_type


def _create_output_directory_if_missing(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    print(f"📁 Saving evaluation outputs to: {path}")


def _validate_required_files(model_path: str, data_path: str) -> bool:
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the probe first using train_probe.py")
        return False
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please make sure you've run the data extraction and processing notebook first.")
        return False
    return True


def _load_split_indices() -> Tuple[List[int], List[int]]:
    """Load split indices from a shared, hardcoded location."""
    split_path = "/content/drive/MyDrive/probes/split_indices.json"
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split indices not found at {split_path}. Run training first to create a deterministic split."
        )
    with open(split_path, "r") as f:
        data = json.load(f)
    return data.get("train_indices", []), data.get("test_indices", [])


def _build_test_loader(data_path: str, feature_type: str) -> Tuple[DataLoader, ProbeDataset, List[int]]:
    """Build test loader using persisted test indices; returns aligned original indices."""
    backbone_features, action_targets = load_probe_data(data_path, feature_type=feature_type)
    _, test_indices = _load_split_indices()

    test_features = [backbone_features[i] for i in test_indices]
    test_targets = [action_targets[i] for i in test_indices]

    valid_test_indices = [
        test_indices[i]
        for i in range(len(test_indices))
        if test_features[i] is not None and test_targets[i] is not None
    ]

    test_dataset = ProbeDataset(test_features, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader, test_dataset, valid_test_indices


def _infer_input_output_dims(dataset: ProbeDataset) -> Tuple[int, int]:
    sample_features, sample_target = dataset[0]
    input_dim = sample_features.shape[-1]
    output_dim = sample_target.shape[-1] if len(sample_target.shape) > 0 else 1
    return input_dim, output_dim


def _plot_artifacts_if_available(
    history_path: str, output_dir: str, predictions: np.ndarray, targets: np.ndarray
) -> Tuple[str, str]:
    training_curves_path = None
    if os.path.exists(history_path):
        print("\n📈 Plotting training curves...")
        training_curves_path = os.path.join(output_dir, "training_curves.png")
        plot_training_history(history_path, save_path=training_curves_path)

    print("\n📊 Plotting predictions vs targets...")
    predictions_path = os.path.join(output_dir, "predictions_vs_targets.png")
    plot_predictions_vs_targets(predictions, targets, save_path=predictions_path)
    return training_curves_path, predictions_path


def _save_first_n_predictions_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    n: int,
    output_dir: str,
    sample_indices: List[int],
) -> str:
    first_n = min(n, predictions.shape[0])
    preds = predictions[:first_n]
    targs = targets[:first_n]

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if targs.ndim == 1:
        targs = targs.reshape(-1, 1)

    num_dims = preds.shape[1]
    csv_headers = (
        ["index", "sample_index"] + [f"pred_{i}" for i in range(num_dims)] + [f"target_{i}" for i in range(num_dims)]
    )
    predictions_csv_path = os.path.join(output_dir, "first_100_predictions.csv")
    with open(predictions_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        for idx in range(first_n):
            row = (
                [idx, sample_indices[idx] if idx < len(sample_indices) else "NA"]
                + preds[idx].tolist()
                + targs[idx].tolist()
            )
            writer.writerow(row)
    return predictions_csv_path


def _save_metrics_pickle(metrics: Dict[str, float], output_dir: str) -> str:
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ["predictions", "targets"]}
    evaluation_metrics_path = os.path.join(output_dir, "evaluation_metrics.pkl")
    with open(evaluation_metrics_path, "wb") as f:
        pickle.dump(metrics_to_save, f)
    return evaluation_metrics_path


def main(feature_type: str = "mean_pooled", data_path: str = None, model_path: str = None):
    """Main evaluation function orchestrating the full probe evaluation pipeline."""
    # Setup
    probe_output_dir, MODEL_PATH, DATA_PATH, HISTORY_PATH, FEATURE_TYPE = _configure_paths(
        feature_type, data_path, model_path
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    print(f"Feature type: {FEATURE_TYPE}")

    # Deterministic seeding for reproducible splits and predictions
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    _create_output_directory_if_missing(probe_output_dir)

    # Validate inputs
    if not _validate_required_files(MODEL_PATH, DATA_PATH):
        return

    # Data
    test_loader, test_dataset, valid_test_indices = _build_test_loader(DATA_PATH, FEATURE_TYPE)
    input_dim, output_dim = _infer_input_output_dims(test_dataset)
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = load_trained_model(MODEL_PATH, input_dim, output_dim)
    print(f"✅ Loaded trained linear regression model from {MODEL_PATH}")

    # Evaluation
    print("\n🔍 Evaluating model...")
    metrics = detailed_evaluation(model, test_loader, device=DEVICE)
    print_evaluation_summary(metrics)

    # Plots
    training_curves_path, predictions_path = _plot_artifacts_if_available(
        HISTORY_PATH, probe_output_dir, metrics["predictions"], metrics["targets"]
    )

    # Artifacts
    predictions_csv_path = _save_first_n_predictions_csv(
        metrics["predictions"],
        metrics["targets"],
        n=100,
        output_dir=probe_output_dir,
        sample_indices=valid_test_indices,
    )
    evaluation_metrics_path = _save_metrics_pickle(metrics, probe_output_dir)

    # Final messages
    print(f"\n💾 Evaluation metrics saved to: {evaluation_metrics_path}")
    if training_curves_path is not None:
        print(f"📈 Training curves saved to: {training_curves_path}")
    print(f"📊 Predictions plot saved to: {predictions_path}")
    print(f"🧾 First 100 predictions saved to: {predictions_csv_path}")
    print(f"📁 All evaluation outputs in: {probe_output_dir}")
    print(f"Feature type used: {FEATURE_TYPE}")
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()
