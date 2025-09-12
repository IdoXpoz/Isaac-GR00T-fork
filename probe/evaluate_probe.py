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

        # Calculate grid size dynamically based on number of dimensions
        n_cols = min(3, n_dims)  # Max 3 columns
        n_rows = (n_dims + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # Handle single subplot case
        if n_dims == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
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

        # Hide unused subplots if any
        total_subplots = n_rows * n_cols
        for i in range(n_dims, total_subplots):
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


def _configure_paths(feature_col_name: str, action_step: int, data_path: str) -> Tuple[str, str, str, str, str]:
    """Return output dir, model path, data path, history path, feature col name label."""
    output_base_dir = "/content/drive/MyDrive/probes"
    probe_output_dir = os.path.join(output_base_dir, feature_col_name, f"action_step_{action_step}")
    model_path_final = os.path.join(probe_output_dir, "best_probe_model.pth")
    data_path_final = data_path or "/content/drive/MyDrive/probes/probe_training_data_60k_processed.parquet"
    history_path = os.path.join(probe_output_dir, "training_history.pkl")
    return probe_output_dir, model_path_final, data_path_final, history_path, feature_col_name


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
    output_base_dir = "/content/drive/MyDrive/probes"
    split_path = os.path.join(output_base_dir, "split_indices.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split indices not found at {split_path}. Run training first to create a deterministic split."
        )
    with open(split_path, "r") as f:
        data = json.load(f)
    return data.get("train_indices", []), data.get("test_indices", [])


def _build_test_loader(
    data_path: str, feature_col_name: str, action_step: int
) -> Tuple[DataLoader, ProbeDataset, List[int]]:
    """Build test loader using persisted test indices; returns aligned original indices."""
    backbone_features, action_targets = load_probe_data(
        data_path, feature_col_name=feature_col_name, action_step=action_step
    )
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


def evaluate_single_probe(feature_col_name: str = "mean_pooled_layer_1", action_step: int = 0, data_path: str = None):
    # Setup
    probe_output_dir, MODEL_PATH, DATA_PATH, HISTORY_PATH, FEATURE_COL_NAME = _configure_paths(
        feature_col_name, action_step, data_path
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    print(f"Feature col name: {FEATURE_COL_NAME}")
    print(f"Action step: {action_step}")

    _create_output_directory_if_missing(probe_output_dir)

    # Validate inputs
    if not _validate_required_files(MODEL_PATH, DATA_PATH):
        return

    # Data
    test_loader, test_dataset, valid_test_indices = _build_test_loader(DATA_PATH, FEATURE_COL_NAME, action_step)
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
    print(f"Feature col name used: {FEATURE_COL_NAME}")
    print(f"Action step used: {action_step}")
    print("🎉 Evaluation completed!")


def evaluate_all_probes_for_single_action_step(
    data_path: str = "/content/drive/MyDrive/probes/probe_training_data_60k_processed.parquet",
    action_step: int = 0,
):
    """Evaluate all probes for a single action step.

    Args:
        data_path: Path to the processed data file
        action_step: Which action step to evaluate (0-based)
    """
    print(f"\n🔍 Evaluating all probes for action step {action_step}")
    print("=" * 60)

    for layer in range(0, 5):
        for pooling in ["mean_pooled", "last_vector"]:
            feature_col_name = f"{pooling}_layer_{layer}"
            print(f"\n📊 Evaluating {feature_col_name} for action step {action_step}")
            print("-" * 40)

            try:
                evaluate_single_probe(
                    feature_col_name=feature_col_name,
                    action_step=action_step,
                    data_path=data_path,
                )
            except Exception as e:
                print(f"❌ Error evaluating {feature_col_name}: {str(e)}")
                continue

    print(f"\n🎉 Completed evaluation of all probes for action step {action_step}")


def compare_all_probes_for_action_step(action_step: int = 0, show_plot: bool = True) -> Dict[str, Dict[str, float]]:
    """Create a comparison graph of MSE and mean correlation for all probes for a single action step.

    Args:
        action_step: Which action step to compare (0-based)
        output_dir: Directory to save the comparison plot. If None, uses default location.
        show_plot: Whether to display the plot

    Returns:
        Dictionary containing metrics for all probes
    """
    print(f"\n📊 Comparing all probes for action step {action_step}")
    print("=" * 60)

    # Define all probe configurations
    layers = list(range(0, 5))
    pooling_methods = ["mean_pooled", "last_vector"]

    # Storage for metrics
    probe_names = []
    mse_values = []
    correlation_values = []
    missing_probes = []

    # Load metrics for each probe
    output_base_dir = "/content/drive/MyDrive/probes"

    for pooling in pooling_methods:
        for layer in layers:
            feature_col_name = f"{pooling}_layer_{layer}"
            probe_output_dir = os.path.join(output_base_dir, feature_col_name, f"action_step_{action_step}")
            metrics_path = os.path.join(probe_output_dir, "evaluation_metrics.pkl")

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "rb") as f:
                        metrics = pickle.load(f)

                    probe_names.append(feature_col_name)
                    mse_values.append(metrics["mse"])

                    # Calculate mean correlation
                    correlations = metrics["correlations"]
                    if isinstance(correlations, list):
                        mean_corr = np.mean([c for c in correlations if not np.isnan(c)])
                    else:
                        mean_corr = correlations if not np.isnan(correlations) else 0.0
                    correlation_values.append(mean_corr)

                    print(f"✅ Loaded {feature_col_name}: MSE={metrics['mse']:.6f}, Mean Corr={mean_corr:.4f}")

                except Exception as e:
                    print(f"❌ Error loading {feature_col_name}: {str(e)}")
                    missing_probes.append(feature_col_name)
            else:
                print(f"⚠️  Missing metrics for {feature_col_name}")
                missing_probes.append(feature_col_name)

    if not probe_names:
        print("❌ No probe metrics found. Make sure to run evaluation first.")
        return {}

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors for different pooling methods
    colors = []
    for name in probe_names:
        if "mean_pooled" in name:
            colors.append("steelblue")
        else:
            colors.append("darkorange")

    # MSE comparison
    bars1 = ax1.bar(range(len(probe_names)), mse_values, color=colors, alpha=0.7)
    ax1.set_xlabel("Probe Configuration")
    ax1.set_ylabel("MSE (Lower is Better)")
    ax1.set_title(f"MSE Comparison - Action Step {action_step}")
    ax1.set_xticks(range(len(probe_names)))
    ax1.set_xticklabels(probe_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, mse_values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mse_values) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Correlation comparison
    bars2 = ax2.bar(range(len(probe_names)), correlation_values, color=colors, alpha=0.7)
    ax2.set_xlabel("Probe Configuration")
    ax2.set_ylabel("Mean Correlation (Higher is Better)")
    ax2.set_title(f"Mean Correlation Comparison - Action Step {action_step}")
    ax2.set_xticks(range(len(probe_names)))
    ax2.set_xticklabels(probe_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, correlation_values)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(correlation_values) * 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.7, label="Mean Pooled"),
        Patch(facecolor="darkorange", alpha=0.7, label="Last Vector"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    output_dir = os.path.join(output_base_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f"probe_comparison_action_step_{action_step}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    # Print summary
    print(f"\n📈 Best performing probes for action step {action_step}:")

    # Find best MSE (lowest)
    if mse_values:
        best_mse_idx = np.argmin(mse_values)
        print(f"  🏆 Lowest MSE: {probe_names[best_mse_idx]} ({mse_values[best_mse_idx]:.6f})")

    # Find best correlation (highest)
    if correlation_values:
        best_corr_idx = np.argmax(correlation_values)
        print(f"  🏆 Highest Correlation: {probe_names[best_corr_idx]} ({correlation_values[best_corr_idx]:.4f})")

    if missing_probes:
        print(f"\n⚠️  Missing evaluations for: {', '.join(missing_probes)}")

    print(f"\n💾 Comparison plot saved to: {plot_path}")

    # Return organized results
    results = {}
    for i, name in enumerate(probe_names):
        results[name] = {"mse": mse_values[i], "mean_correlation": correlation_values[i]}

    return results


def evaluate_all_action_steps_for_specific_layer(
    layer: int,
    pooling_method: str,
    data_path: str = "/content/drive/MyDrive/probes/probe_training_data_60k_processed.parquet",
    max_action_steps: int = 16,
):
    """Evaluate probes for a specific layer and pooling method across all action steps.

    Args:
        layer: Layer number (0-4)
        pooling_method: Pooling method ("mean_pooled" or "last_vector")
        data_path: Path to the processed data file
        max_action_steps: Maximum number of action steps to evaluate
    """
    feature_col_name = f"{pooling_method}_layer_{layer}"

    print(f"\n🔍 Evaluating {feature_col_name} across {max_action_steps} action steps")
    print("=" * 60)

    for action_step in range(max_action_steps):
        print(f"\n📊 Evaluating {feature_col_name} for action step {action_step}")
        print("-" * 40)

        try:
            evaluate_single_probe(
                feature_col_name=feature_col_name,
                action_step=action_step,
                data_path=data_path,
            )
        except Exception as e:
            print(f"❌ Error evaluating action step {action_step}: {str(e)}")
            continue

    print(f"\n🎉 Completed evaluation of {feature_col_name} across all action steps")


def compare_all_action_steps_for_specific_layer(
    layer: int,
    pooling_method: str,
    max_action_steps: int = 16,
    show_plot: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Create a comparison graph of MSE and mean correlation across action steps for a specific layer.

    Args:
        layer: Layer number (0-4)
        pooling_method: Pooling method ("mean_pooled" or "last_vector")
        max_action_steps: Maximum number of action steps to compare
        show_plot: Whether to display the plot

    Returns:
        Dictionary containing metrics for all action steps
    """
    feature_col_name = f"{pooling_method}_layer_{layer}"

    print(f"\n📊 Comparing {feature_col_name} across {max_action_steps} action steps")
    print("=" * 60)

    # Storage for metrics
    action_step_names = []
    mse_values = []
    correlation_values = []
    missing_steps = []

    # Load metrics for each action step
    output_base_dir = "/content/drive/MyDrive/probes"

    for action_step in range(max_action_steps):
        probe_output_dir = os.path.join(output_base_dir, feature_col_name, f"action_step_{action_step}")
        metrics_path = os.path.join(probe_output_dir, "evaluation_metrics.pkl")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "rb") as f:
                    metrics = pickle.load(f)

                action_step_names.append(f"Step {action_step}")
                mse_values.append(metrics["mse"])

                # Calculate mean correlation
                correlations = metrics["correlations"]
                if isinstance(correlations, list):
                    mean_corr = np.mean([c for c in correlations if not np.isnan(c)])
                else:
                    mean_corr = correlations if not np.isnan(correlations) else 0.0
                correlation_values.append(mean_corr)

                print(f"✅ Loaded action step {action_step}: MSE={metrics['mse']:.6f}, Mean Corr={mean_corr:.4f}")

            except Exception as e:
                print(f"❌ Error loading action step {action_step}: {str(e)}")
                missing_steps.append(action_step)
        else:
            print(f"⚠️  Missing metrics for action step {action_step}")
            missing_steps.append(action_step)

    if not action_step_names:
        print("❌ No action step metrics found. Make sure to run evaluation first.")
        return {}

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # MSE comparison
    bars1 = ax1.bar(range(len(action_step_names)), mse_values, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Action Step")
    ax1.set_ylabel("MSE (Lower is Better)")
    ax1.set_title(f"MSE Comparison - {feature_col_name}")
    ax1.set_xticks(range(len(action_step_names)))
    ax1.set_xticklabels(action_step_names, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, mse_values)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mse_values) * 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Correlation comparison
    bars2 = ax2.bar(range(len(action_step_names)), correlation_values, color="darkorange", alpha=0.7)
    ax2.set_xlabel("Action Step")
    ax2.set_ylabel("Mean Correlation (Higher is Better)")
    ax2.set_title(f"Mean Correlation Comparison - {feature_col_name}")
    ax2.set_xticks(range(len(action_step_names)))
    ax2.set_xticklabels(action_step_names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, correlation_values)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(correlation_values) * 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    output_dir = os.path.join(output_base_dir, "comparisons")
    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f"action_steps_comparison_{feature_col_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()

    # Print summary
    print(f"\n📈 Best performing action steps for {feature_col_name}:")

    # Find best MSE (lowest)
    if mse_values:
        best_mse_idx = np.argmin(mse_values)
        print(f"  🏆 Lowest MSE: {action_step_names[best_mse_idx]} ({mse_values[best_mse_idx]:.6f})")

    # Find best correlation (highest)
    if correlation_values:
        best_corr_idx = np.argmax(correlation_values)
        print(f"  🏆 Highest Correlation: {action_step_names[best_corr_idx]} ({correlation_values[best_corr_idx]:.4f})")

    if missing_steps:
        print(f"\n⚠️  Missing evaluations for action steps: {', '.join(map(str, missing_steps))}")

    print(f"\n💾 Comparison plot saved to: {plot_path}")

    # Return organized results
    results = {}
    for i, name in enumerate(action_step_names):
        results[name] = {"mse": mse_values[i], "mean_correlation": correlation_values[i]}

    return results
