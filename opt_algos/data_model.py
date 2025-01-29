import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import plotly.graph_objects as go
import os
import pandas as pd
from typing import Union, Tuple


def load_checkpoint_run(run_dir, device='cpu'):
    """
    Load checkpoint, configuration and feature mask from a specific run directory.

    Args:
        run_dir (str): Path to the run directory containing checkpoints, config and feature mask files
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        tuple: A tuple containing:
            - checkpoint_data (dict): The loaded checkpoint containing model state and training info
            - config_data (dict): Configuration parameters used for the run
            - feature_mask_data (dict): Feature mask indicating which input features were used

    Raises:
        FileNotFoundError: If checkpoint, config or feature mask files don't exist
        RuntimeError: If there are issues loading the checkpoint
    """
    run_path = Path(run_dir)

    # Load latest checkpoint with device mapping
    checkpoint_path = run_path / "checkpoints" / "checkpoint_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    config_path = run_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load feature mask
    feature_mask_path = run_path / "feature_mask.json"
    with open(feature_mask_path, "r") as f:
        feature_mask = json.load(f)

    return checkpoint, config, feature_mask


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, feature_mask):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )
        self.feature_mask = feature_mask

    def forward(self, x):
        # Use the device of the model's parameters
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.layers[0].weight.device)
        elif x.device != self.layers[0].weight.device:
            x = x.to(self.layers[0].weight.device)
        
        # Create mask tensor on the same device as input
        mask = torch.tensor(self.feature_mask, dtype=torch.bool, device=x.device)
        x = x[:, mask]
        return self.layers(x)


def load_model_for_prediction(checkpoint_path, device=None):
    """
    Load a trained model and its normalization statistics from a checkpoint.

    Args:
        checkpoint_path (str): Path to directory containing model checkpoint and config files
        device (str): Device to load the model on

    Returns:
        tuple: A tuple containing:
            - model (MLP): Initialized and loaded model ready for prediction
            - norm_stats (dict): Normalization statistics for input/output scaling
    """
    # Load checkpoint, config and feature mask
    checkpoint, config, feature_mask_data = load_checkpoint_run(checkpoint_path, device=device)

    # Extract feature mask and convert to list of booleans
    feature_mask = feature_mask_data.get("feature_mask", [True] * 9)
    if isinstance(feature_mask, dict):
        feature_mask = [
            bool(feature_mask[str(i)]) if str(i) in feature_mask else True
            for i in range(9)
        ]

    # Calculate actual input size after feature masking
    actual_input_size = sum(feature_mask)  # Count number of True values in mask

    # Initialize model with correct input size
    model = MLP(
        input_size=actual_input_size,  # Use actual number of features after masking
        hidden_size=config["hidden_size"],
        output_size=11,
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"],
        feature_mask=feature_mask,
    )
    
    if device:
        model = model.to(device)
    
    # Load only the layer parameters from the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    
    return model, checkpoint["normalization_stats"]


def predict(model, X, norm_stats=None):
    """
    Make predictions using the model with optional input/output normalization.

    Args:
        model (MLP): The trained model to use for prediction
        X (numpy.ndarray): Input features of shape (batch_size, num_features)
        norm_stats (dict, optional): Normalization statistics containing:
            - X_mean: Mean of input features
            - X_std: Standard deviation of input features
            - y_mean: Mean of target values
            - y_std: Standard deviation of target values
            If None, no normalization is applied. Defaults to None.

    Returns:
        numpy.ndarray: Model predictions of shape (batch_size, num_outputs).
            If norm_stats provided, predictions are denormalized to original scale.

    Note:
        Small epsilon (1e-8) is added to standard deviations to prevent division by zero
    """
    model.eval()
    device = next(model.parameters()).device  # Get model's device
    
    with torch.no_grad():
        # Normalize input if needed
        if norm_stats is not None:
            eps = 1e-8
            X_std = np.where(norm_stats["X_std"] == 0, eps, norm_stats["X_std"])
            X_normalized = (X - norm_stats["X_mean"]) / X_std
            X_tensor = torch.tensor(X_normalized, dtype=torch.float32, device=device)
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        # Get predictions
        y_pred_normalized = model(X_tensor).cpu().numpy()

        # Denormalize predictions if needed
        if norm_stats is not None:
            y_std = np.where(norm_stats["y_std"] == 0, eps, norm_stats["y_std"])
            y_pred = y_pred_normalized * y_std + norm_stats["y_mean"]
        else:
            y_pred = y_pred_normalized

        return y_pred


def get_flops(model_size: Union[str, int, float], step: Union[int, float]) -> float:
    """
    Get the FLOPS value for a given model size and training step.
    
    Args:
        model_size: Size of the model. Can be either:
            - str: e.g., "20M", "300M", "1B"
            - int/float: size in millions of parameters (e.g., 20 for 20M, 1000 for 1B)
        step (Union[int, float]): Training step to get FLOPS for
        
    Returns:
        float: FLOPS value for the given configuration
        
    Raises:
        ValueError: If model_size format is invalid or not found in dataset
        ValueError: If step is not found for the given model_size
        TypeError: If inputs are of invalid types
    """
    # Input validation
    if not isinstance(step, (int, float)):
        raise TypeError(f"step must be int or float, got {type(step)}")
    if not isinstance(model_size, (str, int, float)):
        raise TypeError(f"model_size must be str, int, or float, got {type(model_size)}")
    
    # Convert numeric model_size to string format
    if isinstance(model_size, (int, float)):
        if model_size >= 1000:
            model_size = f"{int(model_size/1000)}B"
        else:
            model_size = f"{int(model_size)}M"
    
    # Validate string format
    if isinstance(model_size, str):
        if not (model_size.endswith('M') or model_size.endswith('B')):
            raise ValueError(f"String model_size must end with 'M' or 'B', got {model_size}")
        try:
            # Extract numeric part and validate
            size_num = float(model_size[:-1])
            if model_size.endswith('B'):
                size_num *= 1000
            if size_num <= 0:
                raise ValueError(f"Model size must be positive, got {size_num}")
        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise ValueError(f"Invalid model size format: {model_size}")
            raise
    
    # Load the FLOPS dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flops_path = os.path.join(script_dir, "flops_dataset.csv")
    df = pd.read_csv(flops_path)
    
    # Filter for the requested model size
    model_df = df[df["model_size"] == model_size]
    if len(model_df) == 0:
        raise ValueError(f"Model size {model_size} not found in dataset. Available sizes: {sorted(df['model_size'].unique())}")
    
    # Find the exact step if it exists
    if step in model_df["step"].values:
        return float(model_df[model_df["step"] == step]["flops"].iloc[0])
    
    # If exact step not found, interpolate between closest steps
    steps = model_df["step"].values
    if step < steps.min() or step > steps.max():
        raise ValueError(f"Step {step} is outside the available range [{steps.min()}, {steps.max()}] for model size {model_size}")
    
    # Find closest steps for interpolation
    lower_step = steps[steps <= step].max()
    upper_step = steps[steps >= step].min()
    
    lower_flops = float(model_df[model_df["step"] == lower_step]["flops"].iloc[0])
    upper_flops = float(model_df[model_df["step"] == upper_step]["flops"].iloc[0])
    
    # Linear interpolation
    ratio = (step - lower_step) / (upper_step - lower_step)
    interpolated_flops = lower_flops + ratio * (upper_flops - lower_flops)
    
    return interpolated_flops


if __name__ == "__main__":
    # NOTE: EXAMPLE USAGE, the above code is modular and can be used in other scripts.

    def map_runs_by_split():
        """
        Scans through all run directories and creates a dictionary mapping
        split types to their corresponding run directories
        """
        runs_dir = Path("runs")
        splits_dict = {}

        # Iterate through all directories in runs/
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "wandb":
                continue

            # Try to load the config file
            config_path = run_dir / "config.json"
            if not config_path.exists():
                continue

            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Get the split type from config
                split_type = config.get("split_type")
                if split_type:
                    if split_type not in splits_dict:
                        splits_dict[split_type] = []
                    splits_dict[split_type].append(str(run_dir))

            except json.JSONDecodeError:
                print(f"Error reading config from {run_dir}")
                continue

        # Print the results in a formatted way
        print("=== Runs by Split Type ===")
        for split_type, runs in splits_dict.items():
            print(f"\n{split_type}:")
            for run in runs:
                print(f"  - {run}")

        return splits_dict

    # Create the mapping
    splits_dict = map_runs_by_split()

    def analyze_token_proportion_smoothness(split_type, num_points=1000):
        """
        Analyzes how smoothly the model responds to changes in token proportions
        by sweeping across one proportion while maintaining sum = 1
        """
        if split_type not in splits_dict:
            raise ValueError(
                f"Invalid split type. Must be one of: {list(splits_dict.keys())}"
            )

        model_path = splits_dict[split_type][0]  # Use first model from the split type
        # checkpoint_path = Path(model_path) / "checkpoints" / "checkpoint_latest.pt"

        # Load model and normalization stats using the modular function
        model, norm_stats = load_model_for_prediction(model_path)
        model.eval()

        # Create sweep values for first proportion (0 to 1)
        sweep_values = np.linspace(0, 1, num_points)

        # Initialize results dictionary
        results = {"sweep_values": sweep_values, "predictions": []}

        # For each sweep value, distribute remaining probability equally
        for p1 in sweep_values:
            # Remaining probability to distribute among other 4 proportions
            remaining = 1 - p1
            if remaining < 0:
                continue

            # Distribute remaining equally among other proportions
            other_props = remaining / 4

            # Create input array [p1, p2, p3, p4, p5, *other_features]
            x = np.zeros(9)  # Assuming 9 total features
            x[0] = p1
            x[1:5] = other_props  # Equal distribution for other proportions

            # Set other features to some reasonable default
            x[5] = 1000  # Model size in millions
            x[6] = 2048  # d_model dimension
            x[7] = 16  # Number of attention heads
            x[8] = 15000  # Training steps

            # Get prediction using the modular predict function
            pred = predict(model, x.reshape(1, -1), norm_stats)
            results["predictions"].append(pred.squeeze())

        # Convert predictions to numpy array
        results["predictions"] = np.array(results["predictions"])

        # Create visualization
        fig = go.Figure()

        # Plot prediction for each output dimension
        datasets = {
            "train_cross_entropy": 0,
            "commoncrawl": 1,
            "c4": 2,
            "wikipedia": 3,
            "stackexchange": 4,
            "github": 5,
            "arxiv": 6,
            "book": 7,
            "hellaswag": 8,
            "piqa": 9,
            "arc_easy": 10,
        }

        for dataset, idx in datasets.items():
            fig.add_trace(
                go.Scatter(
                    x=sweep_values,
                    y=results["predictions"][:, idx],
                    name=dataset,
                    mode="lines",
                )
            )

        fig.update_layout(
            title="Model Predictions vs First Token Proportion (Denormalized)",
            xaxis_title="First Token Proportion",
            yaxis_title="Predicted Value",
            height=600,
            width=1000,
        )

        return fig, results

    split_type = "single_step_15000_split"
    fig, results = analyze_token_proportion_smoothness(split_type)
    fig.show()

