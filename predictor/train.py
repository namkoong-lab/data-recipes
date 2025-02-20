import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import os
from sklearn.metrics import r2_score
import time
import json
from pathlib import Path


class MLP(nn.Module):
    def __init__(
        self,
        input_size=9,
        hidden_size=64,
        output_size=11,
        num_layers=2,
        dropout_rate=0.2,
        feature_mask=None,
    ):
        super(MLP, self).__init__()

        assert hidden_size > 0, "Hidden size must be greater than 0"
        assert num_layers >= 1, "Number of layers must be at least 1"
        assert 0 <= dropout_rate < 1, "Dropout rate must be between 0 and 1"

        self.feature_mask = (
            feature_mask if feature_mask is not None else [True] * input_size
        )
        actual_input_size = sum(self.feature_mask)

        layers = []
        # Input layer
        layers.append(nn.Linear(actual_input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Output layer (no dropout after final layer)
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Convert feature mask to boolean tensor on same device as input
        mask = torch.tensor(self.feature_mask, device=x.device, dtype=torch.bool)
        # Select features using the mask
        x = x[:, mask]
        return self.layers(x)


class ModelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_data(X_train, X_test, y_train, y_test):
    """Normalize training and test data using training statistics"""
    # Compute statistics from training data
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    # Add small epsilon to avoid division by zero
    eps = 1e-8
    X_std = np.where(X_std == 0, eps, X_std)
    y_std = np.where(y_std == 0, eps, y_std)

    # Normalize
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm


def load_split_data(split_type):
    """Load data from the correct subdirectory based on split type"""
    base_dir = "/shared/share_mala/andrew/data-recipes/simulator/data"
    split_dir = os.path.join(base_dir, split_type)

    try:
        X_train = np.load(os.path.join(split_dir, "X_train.npy"))
        X_test = np.load(os.path.join(split_dir, "X_test.npy"))
        y_train = np.load(os.path.join(split_dir, "y_train.npy"))
        y_test = np.load(os.path.join(split_dir, "y_test.npy"))
        return X_train, X_test, y_train, y_test

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not load data for split '{split_type}'. Directory or files missing in: {split_dir}"
        )


def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = ModelDataset(X_train, y_train)
    test_dataset = ModelDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Define individual features and their indices
FEATURES = {
    "token_probs": [0, 1, 2, 3, 4],  # First 5 features (token probabilities)
    "model_arch": [6, 7],  # d_model and n_heads
    "training": [5, 8],  # group size and step
}


def create_feature_mask(feature_groups):
    """
    Create a binary mask from a list of feature group names.

    Args:
        feature_groups: List of feature group names to include

    Returns:
        Binary list of length 9 where 1 indicates feature is used
    """
    mask = [0] * 9  # Initialize mask with all zeros

    # Set 1 for indices in selected feature groups
    for group in feature_groups:
        for idx in FEATURES[group]:
            mask[idx] = 1

    return mask


def get_size_from_index(index):
    """Convert numeric index to model size string"""
    # DEBUG print
    print(f"Converting index {index} to size string")
    try:
        if index >= 1000:  # Convert billions to millions (e.g., 1000 -> '1B')
            return f"{int(index/1000)}B"
        else:  # Keep as millions (e.g., 20 -> '20M')
            return f"{int(index)}M"
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        return "UNKNOWN"


def save_checkpoint(model, optimizer, epoch, run_dir, metrics, config, norm_stats=None):
    """
    Save model checkpoint and training state

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        run_dir: Directory to save checkpoint
        metrics: Dictionary of current metrics
        config: Training configuration
        norm_stats: Dictionary containing normalization statistics
    """
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "normalization_stats": norm_stats,  # Add normalization statistics
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint (overwrite)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save config if not already saved
    config_path = run_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    # Save feature mask if not already saved
    mask_path = run_dir / "feature_mask.json"
    if not mask_path.exists():
        with open(mask_path, "w") as f:
            json.dump(
                {
                    "feature_mask": model.feature_mask,
                    "feature_groups": config["feature_groups"],
                },
                f,
                indent=4,
            )


def train():
    wandb.init(project="datarecipe")
    config = wandb.config

    # Create run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/{timestamp}_{wandb.run.id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save wandb config URL
    with open(run_dir / "wandb_run.txt", "w") as f:
        f.write(f"Run URL: {wandb.run.url}\n")
        f.write(f"Run ID: {wandb.run.id}\n")

    # Create feature mask from selected groups
    feature_mask = create_feature_mask(config["feature_groups"])

    # Create model with feature mask
    model = MLP(
        input_size=9,
        hidden_size=config["hidden_size"],
        output_size=11,
        num_layers=config["num_layers"],
        dropout_rate=config["dropout_rate"],
        feature_mask=feature_mask,
    )

    # Load data based on split type
    X_train, X_test, y_train, y_test = load_split_data(config["split_type"])

    # When normalizing data
    if config["normalize_data"]:
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = normalize_data(
            X_train, X_test, y_train, y_test
        )
        # Store normalization statistics
        norm_stats = {
            "y_mean": y_train.mean(axis=0),
            "y_std": y_train.std(axis=0),
            "X_mean": X_train.mean(axis=0),
            "X_std": X_train.std(axis=0),
        }
    else:
        X_train_norm, X_test_norm = X_train, X_test
        y_train_norm, y_test_norm = y_train, y_test
        norm_stats = None

    # Create dataloaders
    train_loader, test_loader = create_data_loaders(
        X_train_norm, X_test_norm, y_train_norm, y_test_norm, batch_size=config["batch_size"]
    )

    # Set device based on config
    device = torch.device(
        f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    for epoch in tqdm(range(config["epochs"]), desc="Training"):
        model.train()
        
        # DEBUG: Print at start of epoch
        print(f"\n=== Epoch {epoch} ===")
        
        # Pre-allocate tensors for all sizes and metrics
        total_train_losses = {
            size: torch.zeros(11, device=device)
            for size in ["20M", "60M", "150M", "300M", "500M", "700M", "1B"]
        }
        train_counts = {size: 0 for size in total_train_losses.keys()}

        batch_count = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            batch_indices = X_batch[:, 5].long().cpu().numpy()
            batch_sizes = [get_size_from_index(idx) for idx in batch_indices]

            if not batch_sizes:  # Skip batch if no valid sizes
                continue

            optimizer.zero_grad()
            y_pred = model(X_batch)

            # Main loss for backprop
            loss = criterion(y_pred, y_batch)

            # Compute all size masks at once
            size_masks = {
                size: torch.tensor([s == size for s in batch_sizes], device=device)
                for size in total_train_losses.keys()
            }

            # Update losses for all sizes in parallel
            for size, mask in size_masks.items():
                if mask.any():
                    # Compute losses for all metrics at once using masked tensors
                    masked_pred = y_pred[mask]
                    masked_true = y_batch[mask]
                    # Calculate MSE for all metrics in one go
                    size_losses = ((masked_pred - masked_true) ** 2).mean(dim=0)
                    total_train_losses[size] += size_losses
                    train_counts[size] += mask.sum().item()

            loss.backward()
            optimizer.step()

            batch_count += 1

        # DEBUG: Print train counts after training
        print("\nTraining counts per size:")
        for size, count in train_counts.items():
            print(f"{size}: {count}")

        model.eval()
        total_test_losses = {
            size: torch.zeros(11)
            for size in ["20M", "60M", "150M", "300M", "500M", "700M", "1B"]
        }
        test_counts = {size: 0 for size in total_test_losses.keys()}
        
        # Add full test loss tracking
        full_test_loss = torch.zeros(11)
        full_test_count = 0
        
        compute_r2_this_epoch = (
            config["r2_eval_frequency"] is not None
            and epoch % config["r2_eval_frequency"] == 0
        )

        size_predictions = {}
        size_true_values = {}
        full_predictions = []
        full_true_values = []
        
        if compute_r2_this_epoch:
            if config["split_by_size"]:
                size_predictions = {size: [] for size in total_test_losses.keys()}
                size_true_values = {size: [] for size in total_test_losses.keys()}

        # DEBUG: Add batch size tracking for test set
        test_batch_sizes = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                batch_indices = X_batch[:, 5].long().cpu().numpy()
                batch_sizes = [get_size_from_index(idx) for idx in batch_indices]
                
                # DEBUG: Print raw indices and their converted sizes
                print("\nDEBUG - Batch information:")
                print("Raw indices:", batch_indices)
                print("Converted sizes:", batch_sizes)
                
                y_pred = model(X_batch)

                # Collect predictions if we're computing R² this epoch
                if compute_r2_this_epoch:
                    if config["split_by_size"]:
                        for i, size in enumerate(batch_sizes):
                            if size in size_predictions:
                                size_predictions[size].append(y_pred[i].cpu().numpy())
                                size_true_values[size].append(y_batch[i].cpu().numpy())
                    else:
                        full_predictions.append(y_pred.cpu().numpy())
                        full_true_values.append(y_batch.cpu().numpy())

                # Test loss calculation based on split mode
                if config["split_by_size"]:
                    for size in total_test_losses.keys():
                        size_mask = torch.tensor(
                            [s == size for s in batch_sizes], device=device
                        )
                        # DEBUG: Print mask information
                        if size_mask.any():
                            print(f"\nSize {size} mask:")
                            print("Mask:", size_mask)
                            print("Number of True values:", size_mask.sum().item())

                        if size_mask.any():
                            for i in range(11):
                                size_loss = criterion(
                                    y_pred[size_mask, i], y_batch[size_mask, i]
                                )
                                total_test_losses[size][i] += size_loss.item()
                            test_counts[size] += size_mask.sum().item()
                else:  # full mode
                    # Compute loss for full batch
                    for i in range(11):
                        loss = criterion(y_pred[:, i], y_batch[:, i])
                        full_test_loss[i] += loss.item()
                    full_test_count += len(y_batch)

        metrics = {}

        if not config["split_by_size"]:  # Changed from if config["split_by_size"]
            # Compute full dataset metrics
            if full_test_count > 0:
                avg_full_test_losses = full_test_loss / full_test_count
                metrics["full/test_loss"] = avg_full_test_losses.mean().item()
                
                # Individual metric losses
                metrics["full/test_loss/train_cross_entropy"] = avg_full_test_losses[0].item()
                metrics["full/test_loss/commoncrawl"] = avg_full_test_losses[1].item()
                metrics["full/test_loss/c4"] = avg_full_test_losses[2].item()
                metrics["full/test_loss/wikipedia"] = avg_full_test_losses[3].item()
                metrics["full/test_loss/stackexchange"] = avg_full_test_losses[4].item()
                metrics["full/test_loss/github"] = avg_full_test_losses[5].item()
                metrics["full/test_loss/arxiv"] = avg_full_test_losses[6].item()
                metrics["full/test_loss/book"] = avg_full_test_losses[7].item()
                metrics["full/test_loss/hellaswag"] = avg_full_test_losses[8].item()
                metrics["full/test_loss/piqa"] = avg_full_test_losses[9].item()
                metrics["full/test_loss/arc_easy"] = avg_full_test_losses[10].item()

            # Compute R² for full dataset if needed
            if compute_r2_this_epoch:
                y_true_full = np.vstack(full_true_values)
                y_pred_full = np.vstack(full_predictions)
                
                metrics["full/r2/train_cross_entropy"] = r2_score(y_true_full[:, 0], y_pred_full[:, 0])
                metrics["full/r2/commoncrawl"] = r2_score(y_true_full[:, 1], y_pred_full[:, 1])
                metrics["full/r2/c4"] = r2_score(y_true_full[:, 2], y_pred_full[:, 2])
                metrics["full/r2/wikipedia"] = r2_score(y_true_full[:, 3], y_pred_full[:, 3])
                metrics["full/r2/stackexchange"] = r2_score(y_true_full[:, 4], y_pred_full[:, 4])
                metrics["full/r2/github"] = r2_score(y_true_full[:, 5], y_pred_full[:, 5])
                metrics["full/r2/arxiv"] = r2_score(y_true_full[:, 6], y_pred_full[:, 6])
                metrics["full/r2/book"] = r2_score(y_true_full[:, 7], y_pred_full[:, 7])
                metrics["full/r2/hellaswag"] = r2_score(y_true_full[:, 8], y_pred_full[:, 8])
                metrics["full/r2/piqa"] = r2_score(y_true_full[:, 9], y_pred_full[:, 9])
                metrics["full/r2/arc_easy"] = r2_score(y_true_full[:, 10], y_pred_full[:, 10])

        else:  # by_size mode
            for size in total_train_losses.keys():
                if train_counts[size] > 0:
                    avg_train_losses = total_train_losses[size] / train_counts[size]
                    metrics[f"{size}/train_loss"] = avg_train_losses.mean().item()

                if test_counts[size] > 0:
                    avg_test_losses = total_test_losses[size] / test_counts[size]
                    metrics[f"{size}/test_loss"] = avg_test_losses.mean().item()

                    # Test loss logging for individual metrics
                    metrics[f"{size}/test_loss/train_cross_entropy"] = avg_test_losses[0].item()
                    metrics[f"{size}/test_loss/commoncrawl"] = avg_test_losses[1].item()
                    metrics[f"{size}/test_loss/c4"] = avg_test_losses[2].item()
                    metrics[f"{size}/test_loss/wikipedia"] = avg_test_losses[3].item()
                    metrics[f"{size}/test_loss/stackexchange"] = avg_test_losses[4].item()
                    metrics[f"{size}/test_loss/github"] = avg_test_losses[5].item()
                    metrics[f"{size}/test_loss/arxiv"] = avg_test_losses[6].item()
                    metrics[f"{size}/test_loss/book"] = avg_test_losses[7].item()
                    metrics[f"{size}/test_loss/hellaswag"] = avg_test_losses[8].item()
                    metrics[f"{size}/test_loss/piqa"] = avg_test_losses[9].item()
                    metrics[f"{size}/test_loss/arc_easy"] = avg_test_losses[10].item()

                    # Compute size-specific R² if needed
                    if compute_r2_this_epoch:
                        if size_predictions[size]:
                            y_true_size = np.stack(size_true_values[size])
                            y_pred_size = np.stack(size_predictions[size])

                            metrics[f"{size}/r2/train_cross_entropy"] = r2_score(
                                y_true_size[:, 0], y_pred_size[:, 0]
                            )
                            metrics[f"{size}/r2/commoncrawl"] = r2_score(
                                y_true_size[:, 1], y_pred_size[:, 1]
                            )
                            metrics[f"{size}/r2/c4"] = r2_score(
                                y_true_size[:, 2], y_pred_size[:, 2]
                            )
                            metrics[f"{size}/r2/wikipedia"] = r2_score(
                                y_true_size[:, 3], y_pred_size[:, 3]
                            )
                            metrics[f"{size}/r2/stackexchange"] = r2_score(
                                y_true_size[:, 4], y_pred_size[:, 4]
                            )
                            metrics[f"{size}/r2/github"] = r2_score(
                                y_true_size[:, 5], y_pred_size[:, 5]
                            )
                            metrics[f"{size}/r2/arxiv"] = r2_score(
                                y_true_size[:, 6], y_pred_size[:, 6]
                            )
                            metrics[f"{size}/r2/book"] = r2_score(
                                y_true_size[:, 7], y_pred_size[:, 7]
                            )
                            metrics[f"{size}/r2/hellaswag"] = r2_score(
                                y_true_size[:, 8], y_pred_size[:, 8]
                            )
                            metrics[f"{size}/r2/piqa"] = r2_score(
                                y_true_size[:, 9], y_pred_size[:, 9]
                            )
                            metrics[f"{size}/r2/arc_easy"] = r2_score(
                                y_true_size[:, 10], y_pred_size[:, 10]
                            )

        metrics["epoch"] = epoch

        # DEBUG: Print test information
        print("\nTest set information:")
        print("Unique sizes in test set:", set(test_batch_sizes))
        print("\nTest counts per size:")
        for size, count in test_counts.items():
            print(f"{size}: {count}")
            
        if compute_r2_this_epoch:
            print("\nR² computation information:")
            if config["split_by_size"]:
                print("Sizes with predictions:", [size for size, preds in size_predictions.items() if preds])
                for size, preds in size_predictions.items():
                    if preds:
                        print(f"{size}: {len(preds)} predictions")
            else:
                print(f"Full predictions count: {len(full_predictions)}")

        # DEBUG: Print metrics being logged
        print("\nMetrics being logged:")
        for key in metrics.keys():
            if key != "epoch":
                print(f"- {key}")

        # Log all metrics
        wandb.log(metrics)

        # Save intermediate checkpoints based on frequency
        if (
            config.get("save_checkpoint_frequency")
            and config["save_checkpoint_frequency"] not in [None, 0]
            and epoch % config["save_checkpoint_frequency"] == 0
        ):
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                run_dir=run_dir,
                metrics=metrics,
                config=wandb.config._items,
                norm_stats=norm_stats,
            )

    # Save final checkpoint outside the training loop
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"] - 1,  # Zero-based indexing
        run_dir=run_dir,
        metrics=metrics,
        config=wandb.config._items,
        norm_stats=norm_stats,
    )


if __name__ == "__main__":
    # Update default configuration
    config = {
        "split_type": "step",
        "normalize_data": True,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 25,
        "device": 6,
        "feature_groups": ["token_probs", "training"],
        "r2_eval_frequency": None,
        "save_checkpoint_frequency": 5,  # Save every 5 epochs
    }
    wandb.init(config=config)
    train()
