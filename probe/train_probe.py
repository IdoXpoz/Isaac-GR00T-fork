#!/usr/bin/env python3
"""
GR00T Probe Training Script

This script trains a probe to predict robot action tokens from VLM backbone features.
The probe aims to understand if the VLM's intermediate representations contain
information predictive of the final action outputs.

Author: Generated for GR00T probe analysis
"""

import os
import pickle
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ProbeDataset(Dataset):
    """Dataset class for probe training data."""

    def __init__(self, backbone_features: List[torch.Tensor], action_targets: List[torch.Tensor]):
        """
        Initialize dataset.

        Args:
            backbone_features: List of backbone feature tensors
            action_targets: List of action target tensors
        """
        self.backbone_features = backbone_features
        self.action_targets = action_targets

        # Filter out None values
        valid_indices = [
            i
            for i in range(len(backbone_features))
            if backbone_features[i] is not None and action_targets[i] is not None
        ]

        self.backbone_features = [backbone_features[i] for i in valid_indices]
        self.action_targets = [action_targets[i] for i in valid_indices]

        print(f"Dataset initialized with {len(self.backbone_features)} valid samples")

    def __len__(self) -> int:
        return len(self.backbone_features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample."""
        features = self.backbone_features[idx]
        target = self.action_targets[idx]

        # Ensure tensors are on CPU and properly shaped
        if isinstance(features, torch.Tensor):
            features = features.cpu().float()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        if isinstance(target, torch.Tensor):
            target = target.cpu().float()
        else:
            target = torch.tensor(target, dtype=torch.float32)

        # Features are already mean-pooled to [2048] in load_probe_data function
        # No additional shape handling needed here

        return features, target


class ActionProbe(nn.Module):
    """
    Linear regression probe to predict action tokens from VLM backbone features.

    Architecture:
    - Input: VLM backbone features [2048] (mean pooled or last vector)
    - Linear: Direct linear mapping for regression
    - Output: Action predictions [action_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """
        Initialize the probe.

        Args:
            input_dim: Input feature dimension (e.g., 2048 for backbone features)
            output_dim: Output action dimension
        """
        super().__init__()

        # Simple linear regression layer
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features [batch_size, hidden_size] or [hidden_size]

        Returns:
            Action predictions [batch_size, action_dim] or [action_dim]
        """
        return self.linear(features)


class ProbeTrainer:
    """Handles training and evaluation of the probe model."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize trainer.

        Args:
            model: The probe model to train
            device: Device to run training on
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()

        # Optimizer - Adam with weight decay
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=5, verbose=True
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, targets in tqdm(dataloader, desc="Training"):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss, "mse": avg_loss}

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100, early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the probe model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary containing training history
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]

            # Update history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "probe/best_probe_model.pth")
            else:
                patience_counter += 1

            print(f"Epoch {epoch+1}/{num_epochs}: " f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return history


def load_probe_data(data_path: str, feature_type: str = "mean_pooled") -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Load probe training data from processed parquet file.

    Args:
        data_path: Path to the parquet file containing processed probe data
        feature_type: Type of features to use - "mean_pooled" or "last_vector"

    Returns:
        Tuple of (backbone_features, action_targets)
    """
    print(f"Loading data from {data_path}...")
    print(f"Using feature type: {feature_type}")

    # Load parquet file
    df = pd.read_parquet(data_path)
    print(f"Loaded DataFrame with {len(df)} rows and columns: {list(df.columns)}")

    # Determine which column to use for features
    if feature_type == "mean_pooled":
        feature_column = "backbone_features_mean_pooled"
    elif feature_type == "last_vector":
        feature_column = "backbone_features_last_vector"
    else:
        raise ValueError(f"Invalid feature_type: {feature_type}. Must be 'mean_pooled' or 'last_vector'")

    if feature_column not in df.columns:
        raise ValueError(f"Column '{feature_column}' not found in data. Available columns: {list(df.columns)}")

    backbone_features = []
    action_targets = []

    print(f"Processing {feature_column} and action targets...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
        # Process backbone features
        if row[feature_column] is not None:
            # Features are already processed as [2048] vectors
            feature_array = np.array(row[feature_column])
            backbone_tensor = torch.tensor(feature_array, dtype=torch.float32)
            
            # Ensure shape is [2048]
            if backbone_tensor.shape != torch.Size([2048]):
                print(f"⚠️  Unexpected feature shape: {backbone_tensor.shape}, expected [2048]")
                backbone_features.append(None)
            else:
                backbone_features.append(backbone_tensor)
        else:
            backbone_features.append(None)

        # Process action targets
        if row["action_right_arm_first"] is not None:
            action_tensor = torch.tensor(row["action_right_arm_first"], dtype=torch.float32)
            action_targets.append(action_tensor)
        else:
            action_targets.append(None)

    print(f"Loaded {len(backbone_features)} backbone features")
    print(f"Loaded {len(action_targets)} action targets")

    # Check feature shapes
    valid_features = [f for f in backbone_features if f is not None]
    valid_targets = [t for t in action_targets if t is not None]

    if valid_features:
        print(f"Sample backbone feature shape: {valid_features[0].shape}")
    if valid_targets:
        print(f"Sample action target shape: {valid_targets[0].shape}")

    return backbone_features, action_targets


def split_data(
    backbone_features: List[torch.Tensor], action_targets: List[torch.Tensor], train_ratio: float = 0.99
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Split data into train and test sets.

    Args:
        backbone_features: List of backbone feature tensors
        action_targets: List of action target tensors
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_features, train_targets, test_features, test_targets)
    """
    # Create indices and shuffle
    indices = list(range(len(backbone_features)))
    random.shuffle(indices)

    # Calculate split point
    split_point = int(len(indices) * train_ratio)

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    # Split data
    train_features = [backbone_features[i] for i in train_indices]
    train_targets = [action_targets[i] for i in train_indices]
    test_features = [backbone_features[i] for i in test_indices]
    test_targets = [action_targets[i] for i in test_indices]

    print(f"Split data: {len(train_features)} train, {len(test_features)} test samples")

    return train_features, train_targets, test_features, test_targets


def main(feature_type: str = "mean_pooled", data_path: str = None, batch_size: int = 32, num_epochs: int = 100):
    """Main training function.
    
    Args:
        feature_type: Type of features to use - "mean_pooled" or "last_vector"
        data_path: Path to the processed data file (optional)
        batch_size: Batch size for training
        num_epochs: Number of training epochs
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    DATA_PATH = data_path or "probe_training_data_150k_processed.parquet"  # Use processed data
    FEATURE_TYPE = feature_type
    BATCH_SIZE = batch_size
    NUM_EPOCHS = num_epochs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    print(f"Feature type: {FEATURE_TYPE}")

    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Please make sure you've run the data extraction and processing notebook first.")
        print("The processed file should contain 'backbone_features_mean_pooled' and 'backbone_features_last_vector' columns.")
        return

    backbone_features, action_targets = load_probe_data(DATA_PATH, feature_type=FEATURE_TYPE)

    # Split data
    train_features, train_targets, test_features, test_targets = split_data(
        backbone_features, action_targets, train_ratio=0.99
    )

    # Create datasets
    train_dataset = ProbeDataset(train_features, train_targets)
    test_dataset = ProbeDataset(test_features, test_targets)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get dimensions from sample data
    sample_features, sample_target = train_dataset[0]
    input_dim = sample_features.shape[-1]  # Should be 2048
    output_dim = sample_target.shape[-1] if len(sample_target.shape) > 0 else 1

    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")

    # Create linear regression model (no hidden layers)
    model = ActionProbe(
        input_dim=input_dim,
        output_dim=output_dim,
    )

    print(f"Model architecture:\n{model}")
    print("📊 Using linear regression (no hidden layers)")

    # Create trainer
    trainer = ProbeTrainer(model, device=DEVICE)

    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=15,
    )

    # Final evaluation
    final_metrics = trainer.evaluate(test_loader)
    print(f"\n🎉 Training completed!")
    print(f"Final test loss: {final_metrics['loss']:.6f}")
    print(f"Final test MSE: {final_metrics['mse']:.6f}")

    # Save training history
    with open("probe/training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"Model saved to: probe/best_probe_model.pth")
    print(f"Training history saved to: probe/training_history.pkl")
    print(f"Feature type used: {FEATURE_TYPE}")


if __name__ == "__main__":
    main()
