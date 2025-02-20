import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import yaml
import os
import pickle
import random

class MLP(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=11, num_layers=2, dropout_rate=0.2, feature_mask=None):
        super(MLP, self).__init__()
        
        # Convert feature_mask to boolean list if provided
        self.feature_mask = feature_mask if feature_mask is not None else [True] * input_size
        actual_input_size = sum(self.feature_mask)  # Count number of True values
        
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
        
        # Output layer
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
    
    # Normalize
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm

def load_split_data(split_type):
    """Load data from the correct subdirectory based on split type"""
    base_dir = '/shared/share_mala/andrew/data-recipes/simulator/data'
    split_dir = os.path.join(base_dir, split_type)
    
    try:
        X_train = np.load(os.path.join(split_dir, 'X_train.npy'))
        X_test = np.load(os.path.join(split_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(split_dir, 'y_train.npy'))
        y_test = np.load(os.path.join(split_dir, 'y_test.npy'))
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load data for split '{split_type}'. Directory or files missing in: {split_dir}")

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = ModelDataset(X_train, y_train)
    test_dataset = ModelDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Define individual features and their indices
FEATURES = {
    'token_probs': [0, 1, 2, 3, 4],     # First 5 features (token probabilities)
    'model_arch': [6, 7],               # d_model and n_heads
    'training': [5, 8]                  # group size and step
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

def train():
    wandb.init(project='datarecipe')
    config = wandb.config
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create feature mask from selected groups
    feature_mask = create_feature_mask(config['feature_groups'])
    
    # Create model with feature mask
    model = MLP(
        input_size=9,
        hidden_size=config['hidden_size'],
        output_size=11,
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        feature_mask=feature_mask
    )
    
    # Load data based on split type
    X_train, X_test, y_train, y_test = load_split_data(config['split_type'])
    
    # Optionally normalize data
    if config['normalize_data']:
        X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)
    
    # Create dataloaders
    train_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test, 
        batch_size=config['batch_size']
    )
    
    # Set device based on config
    device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    for epoch in tqdm(range(config['epochs']), desc='Training'):
        # Training
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        total_losses = torch.zeros(11)  # One for each output dimension
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                
                # Calculate loss for each output dimension separately
                for i in range(11):
                    loss = criterion(y_pred[:, i], y_batch[:, i])
                    total_losses[i] += loss.item()
        
        # Average the losses
        avg_losses = total_losses / len(test_loader)
        
        # Log metrics with specific names
        metrics = {
            "train_loss": avg_train_loss,
            "test_loss": avg_losses.mean().item(),  # Overall test loss
            "test_loss/train_cross_entropy": avg_losses[0].item(),
            "test_loss/commoncrawl": avg_losses[1].item(),
            "test_loss/c4": avg_losses[2].item(),
            "test_loss/wikipedia": avg_losses[3].item(),
            "test_loss/stackexchange": avg_losses[4].item(),
            "test_loss/github": avg_losses[5].item(),
            "test_loss/arxiv": avg_losses[6].item(),
            "test_loss/book": avg_losses[7].item(),
            "test_loss/hellaswag": avg_losses[8].item(),
            "test_loss/piqa": avg_losses[9].item(),
            "test_loss/arc_easy": avg_losses[10].item(),
            "epoch": epoch,
        }
        
        wandb.log(metrics)

if __name__ == "__main__":
    # Default configuration
    config = {
        'split_type': 'step',
        'normalize_data': True,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 25,
        'device': 6,
        'feature_groups': ['token_probs', 'training']  # Example: use token probs and training features
    }
    wandb.init(config=config)
    train()