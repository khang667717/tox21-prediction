
#!/usr/bin/env python3
"""
Descriptor-based MLP model for Tox21.
Uses RDKit molecular descriptors as features.
"""
import warnings

from torch_geometric.nn.aggr import scaler
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import json
import argparse
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
import seaborn as sns
from tqdm import tqdm
import sys

from scripts.train_utils import MaskedBCELoss, compute_metrics, set_seed

warnings.filterwarnings('ignore')
set_seed(42)

class DescriptorCalculator:
    """Calculate molecular descriptors using RDKit."""
    
    def __init__(self, descriptor_list: List[str] = None):
        """
        Initialize descriptor calculator.
        
        Args:
            descriptor_list: List of descriptor names to compute.
                           If None, uses all available descriptors.
        """
        if descriptor_list is None:
            # Use all RDKit descriptors
            self.descriptor_list = [desc[0] for desc in Descriptors._descList]
        else:
            self.descriptor_list = descriptor_list
        
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(self.descriptor_list)
    
    def compute_descriptors(self, smiles_list: List[str]) -> np.ndarray:
        """
        Compute descriptors for a list of SMILES.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            numpy array of shape (n_samples, n_descriptors)
        """
        descriptors = []
        invalid_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Computing descriptors")):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_indices.append(i)
                descriptors.append([np.nan] * len(self.descriptor_list))
            else:
                try:
                    desc_vals = self.calculator.CalcDescriptors(mol)
                    descriptors.append(list(desc_vals))
                except:
                    invalid_indices.append(i)
                    descriptors.append([np.nan] * len(self.descriptor_list))
        
        print(f"Computed descriptors for {len(smiles_list)} molecules, {len(invalid_indices)} invalid.")
        
        return np.array(descriptors, dtype=float)
    
    def get_descriptor_names(self) -> List[str]:
        """Get list of descriptor names."""
        return self.descriptor_list

class MLPModel(nn.Module):
    """Multi-layer Perceptron for multi-task classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout_rate: float = 0.2, activation: str = 'relu'):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Input dimension (number of descriptors)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of tasks)
            dropout_rate: Dropout rate
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'tanh')
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

class DescriptorPreprocessor:
    """Preprocess molecular descriptors.
    FIXED: Handle transform on empty array.
    """
    
    def __init__(self, feature_selection_threshold: float = 0.01):
        """
        Initialize preprocessor.
        
        Args:
            feature_selection_threshold: Variance threshold for feature selection
        """
        self.imputer = SimpleImputer(strategy='median')
        self.variance_threshold = VarianceThreshold(threshold=feature_selection_threshold)
        self.scaler = StandardScaler()
        self.selected_features = None
        self.n_features_in_ = None  # FIX: Store number of features
        
    def fit(self, X: np.ndarray, descriptor_names: List[str]):
        """Fit preprocessor on training data."""
        # Handle infinite values
        X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Store number of features after imputation
        self.n_features_in_ = X_imputed.shape[1]
        
        # Check for any remaining infinite values
        if np.any(np.isinf(X_imputed)):
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Remove low variance features
        X_var = self.variance_threshold.fit_transform(X_imputed)
        self.selected_features = self.variance_threshold.get_support()
        
        # Scale features
        _ = self.scaler.fit_transform(X_var)
        
        # Get selected descriptor names
        self.selected_descriptor_names = [
            name for name, selected in zip(descriptor_names, self.selected_features) 
            if selected
        ]
        
        print(f"Original descriptors: {len(descriptor_names)}")
        print(f"After variance threshold: {len(self.selected_descriptor_names)}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor.
        FIXED: Handle empty input array.
        """
        # FIX: Handle empty array
        if X.size == 0:
            if self.n_features_in_ is None:
                raise RuntimeError("Transformer not fitted yet. Cannot determine feature dimension.")
            return np.empty((0, len(self.selected_descriptor_names)))
        
        # Handle infinite values
        X = np.nan_to_num(X, nan=np.nan, posinf=1e10, neginf=-1e10)
        
        X_imputed = self.imputer.transform(X)
        
        # Check for any remaining infinite values
        if np.any(np.isinf(X_imputed)):
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Apply feature selection if fitted
        if self.selected_features is not None:
            X_selected = X_imputed[:, self.selected_features]
        else:
            X_selected = X_imputed
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        return X_scaled
    
    def get_selected_descriptor_names(self) -> List[str]:
        """Get names of selected descriptors."""
        return self.selected_descriptor_names

    
def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray, mask_val: np.ndarray,
                       batch_size: int = 32, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """Create PyTorch DataLoaders for training and validation with feature scaling."""
    
    from sklearn.preprocessing import StandardScaler
    
    # FIX: Scale features to prevent huge validation loss
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    mask_train_tensor = torch.FloatTensor(mask_train)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    mask_val_tensor = torch.FloatTensor(mask_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, mask_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, mask_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle_train, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, val_loader, scaler  # FIX: Return scaler for test set


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in train_loader:
        X_batch, y_batch, mask_batch = batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch, mask_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0

def validate_epoch(model: nn.Module, val_loader: DataLoader, 
                   criterion: nn.Module, device: torch.device,
                   threshold: float = 0.5) -> Tuple[float, Dict[str, Any], np.ndarray]:
    """Validate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    all_masks = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in val_loader:
            X_batch, y_batch, mask_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch, mask_batch)
            
            # Get predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).float()
            
            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_masks.append(mask_batch.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    all_masks = np.vstack(all_masks)
    all_probabilities = np.vstack(all_probabilities)
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions, 
                            all_probabilities, all_masks, threshold)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    return avg_loss, metrics, all_probabilities   

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                device: torch.device, n_epochs: int = 100,
                patience: int = 10, threshold: float = 0.5,
                checkpoint_path: str = None) -> Dict[str, Any]:
    """Train model with early stopping."""
    
    best_val_loss = float('inf')
    best_val_metric = 0  # Using macro PR-AUC as primary metric
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics, _ = validate_epoch(model, val_loader, criterion, 
                                         device, threshold)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Check for improvement (using macro PR-AUC as primary metric)
        val_metric = val_metrics['macro_pr_auc']
        
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint if path provided
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metric': best_val_metric,
                    'val_loss': best_val_loss,
                }, checkpoint_path)
            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{n_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val PR-AUC: {val_metric:.4f}")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest model at epoch {best_epoch + 1}:")
    print(f"  Val Loss: {best_val_loss:.4f}")
    print(f"  Val PR-AUC: {best_val_metric:.4f}")
    
    history['best_epoch'] = best_epoch
    history['best_val_metric'] = best_val_metric
    history['best_val_loss'] = best_val_loss
    
    return history

def plot_training_history(history: Dict[str, Any], save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].axvline(x=history['best_epoch'], color='r', linestyle='--', 
                   label=f'Best Epoch ({history["best_epoch"] + 1})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics
    pr_aucs = [m['macro_pr_auc'] for m in history['val_metrics']]
    roc_aucs = [m['macro_roc_auc'] for m in history['val_metrics']]
    
    axes[1].plot(pr_aucs, label='PR-AUC')
    axes[1].plot(roc_aucs, label='ROC-AUC')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()

def analyze_descriptors(X_train: np.ndarray, descriptor_names: List[str], 
                       y_train: np.ndarray, mask_train: np.ndarray,
                       n_top: int = 20, save_path: str = None):
    """Analyze descriptor importance and correlation."""
    
    # Calculate correlation with each task
    n_tasks = y_train.shape[1]
    task_correlations = []
    
    for task_idx in range(n_tasks):
        task_mask = mask_train[:, task_idx].astype(bool)
        if task_mask.sum() > 10:  # Need enough samples
            y_task = y_train[task_mask, task_idx]
            X_task = X_train[task_mask]
            
            # Calculate correlation for each descriptor
            corrs = []
            for desc_idx in range(X_task.shape[1]):
                # Skip if descriptor is constant
                if np.std(X_task[:, desc_idx]) > 0:
                    corr = np.corrcoef(X_task[:, desc_idx], y_task)[0, 1]
                    corrs.append(abs(corr) if not np.isnan(corr) else 0)
                else:
                    corrs.append(0)
            
            task_correlations.append(corrs)
        else:
            task_correlations.append([0] * X_train.shape[1])
    
    # Average correlation across tasks
    avg_correlations = np.mean(task_correlations, axis=0)
    
    # Get top N descriptors
    top_indices = np.argsort(avg_correlations)[-n_top:][::-1]
    top_descriptors = [descriptor_names[i] for i in top_indices]
    top_correlations = avg_correlations[top_indices]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top descriptors bar plot
    y_pos = np.arange(len(top_descriptors))
    axes[0].barh(y_pos, top_correlations, color='skyblue')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_descriptors)
    axes[0].set_xlabel('Average Absolute Correlation with Tasks')
    axes[0].set_title(f'Top {n_top} Most Correlated Descriptors')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Descriptor correlation matrix (top descriptors)
    X_top = X_train[:, top_indices]
    corr_matrix = np.corrcoef(X_top.T)
    
    im = axes[1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(top_descriptors)))
    axes[1].set_yticks(range(len(top_descriptors)))
    axes[1].set_xticklabels(top_descriptors, rotation=90, fontsize=8)
    axes[1].set_yticklabels(top_descriptors, fontsize=8)
    axes[1].set_title('Descriptor Correlation Matrix')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved descriptor analysis to {save_path}")
    
    plt.show()
    
    return top_descriptors, top_correlations

def main():
    parser = argparse.ArgumentParser(description='Train descriptor-based MLP for Tox21')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='models/descriptor_mlp',
                       help='Directory to save model and results')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer dimensions')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--variance_threshold', type=float, default=0.01,
                       help='Variance threshold for feature selection')
    parser.add_argument('--use_custom_descriptors', action='store_true',
                       help='Use custom descriptor subset')
    parser.add_argument('--tuned_params', type=str, default=None,
                       help='Path to tuned parameters JSON file')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use positive class weights to handle imbalance')
    args = parser.parse_args()
    
    # Load tuned parameters if provided
    tuned_params = None
    if args.tuned_params:
        try:
            with open(args.tuned_params, 'r') as f:
                tuning_results = json.load(f)
                if 'best_params' in tuning_results:
                    tuned_params = tuning_results['best_params']
                elif isinstance(tuning_results, dict):
                    tuned_params = tuning_results
            print(f"✅ Loaded tuned parameters from {args.tuned_params}")
            
            # Override hyperparameters with tuned values
            if 'hidden_dims' in tuned_params:
                hidden_dims = tuned_params['hidden_dims']
                if isinstance(hidden_dims, list):
                    args.hidden_dims = hidden_dims
                else:
                    args.hidden_dims = []
                    n_layers = tuned_params.get('n_layers', 3)
                    for i in range(n_layers):
                        key = f'hidden_dim_{i}'
                        if key in tuned_params:
                            args.hidden_dims.append(tuned_params[key])
                        elif 'hidden_dim' in tuned_params:
                            args.hidden_dims.append(tuned_params['hidden_dim'])
            
            for param in ['dropout_rate', 'learning_rate', 'batch_size', 'weight_decay', 'use_class_weights']:
                if param in tuned_params:
                    setattr(args, param, tuned_params[param])
                    
        except Exception as e:
            print(f"⚠️  Could not load tuned parameters: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_dir = Path(args.data_dir)
    
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    
    # FIX: Check if test set exists and is not empty
    test_path = data_dir / 'test.csv'
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        print(f"Test set loaded: {len(test_df)} samples")
    else:
        test_df = pd.DataFrame()
        print("⚠️ Test set file not found. Creating empty test set.")
    
    train_mask = np.load(data_dir / 'train_mask.npy')
    val_mask = np.load(data_dir / 'val_mask.npy')
    test_mask_path = data_dir / 'test_mask.npy'
    if test_mask_path.exists():
        test_mask = np.load(test_mask_path)
    else:
        test_mask = np.empty((0, 12))
    
    with open(data_dir / 'tasks.json', 'r') as f:
        tasks = json.load(f)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Tasks: {len(tasks)}")
    
    # Prepare targets
    def prepare_targets(df, tasks):
        if len(df) == 0:
            return np.empty((0, len(tasks)))
        targets = df[tasks].values
        return np.nan_to_num(targets, nan=0.0)
    
    y_train = prepare_targets(train_df, tasks)
    y_val = prepare_targets(val_df, tasks)
    y_test = prepare_targets(test_df, tasks)
    
    # Compute descriptors
    print("\nComputing molecular descriptors...")
    
    if args.use_custom_descriptors:
        custom_descriptors = [
            'MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors',
            'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumValenceElectrons',
            'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings',
            'RingCount', 'TPSA', 'LabuteASA', 'BalabanJ', 'BertzCT',
            'FractionCSP3', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3',
            'Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Chi2n', 'Chi3n', 'Chi4n',
            'ExactMolWt', 'HeavyAtomMolWt', 'NHOHCount', 'NOCount',
            'NumRadicalElectrons', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
            'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles'
        ]
        calculator = DescriptorCalculator(custom_descriptors)
    else:
        calculator = DescriptorCalculator()
    
    # Compute descriptors for all splits
    X_train_raw = calculator.compute_descriptors(train_df['canonical_smiles'].tolist())
    X_val_raw = calculator.compute_descriptors(val_df['canonical_smiles'].tolist())
    
    # FIX: Only compute test descriptors if test set has data
    if len(test_df) > 0:
        X_test_raw = calculator.compute_descriptors(test_df['canonical_smiles'].tolist())
    else:
        X_test_raw = np.empty((0, len(calculator.get_descriptor_names())))
        print("Test set is empty, skipping test descriptor computation.")
    
    descriptor_names = calculator.get_descriptor_names()
    print(f"Computed {len(descriptor_names)} descriptors")
    
    # Preprocess descriptors
    print("\nPreprocessing descriptors...")
    preprocessor = DescriptorPreprocessor(feature_selection_threshold=args.variance_threshold)
    
    # Fit on training data only
    preprocessor.fit(X_train_raw, descriptor_names)
    
    # Transform all splits
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)  # FIX: Now handles empty array
    
    print(f"Final feature dimension: {X_train.shape[1]}")
    
    # Save preprocessor and descriptor names
    preprocessor_path = output_dir / 'descriptor_preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    selected_descriptors = preprocessor.get_selected_descriptor_names()
    descriptors_path = output_dir / 'selected_descriptors.json'
    with open(descriptors_path, 'w') as f:
        json.dump(selected_descriptors, f, indent=2)
    
    print(f"Saved preprocessor to {preprocessor_path}")
    print(f"Saved selected descriptors to {descriptors_path}")
    
    # Analyze descriptors (optional but informative)
    if X_train.shape[0] > 0 and y_train.shape[0] > 0:
        print("\nAnalyzing descriptors...")
        analysis_path = output_dir / 'descriptor_analysis.png'
        top_descriptors, top_correlations = analyze_descriptors(
            X_train, selected_descriptors, y_train, train_mask,
            n_top=20, save_path=str(analysis_path)
        )
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, scaler = create_data_loaders(  # FIX: Thêm scaler
        X_train, y_train, train_mask,
        X_val, y_val, val_mask,
        batch_size=args.batch_size
    )

# FIX: Transform test set với scaler đã fit từ training data
    X_test_scaled = scaler.transform(X_test)  # Dùng scaler vừa nhận được
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # Initialize model
    input_dim = X_train.shape[1]
    output_dim = len(tasks)
    
    print(f"\nInitializing MLP model...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimensions: {args.hidden_dims}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Tuned parameters used: {tuned_params is not None}")
    
    model = MLPModel(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=output_dim,
        dropout_rate=args.dropout_rate,
        activation='relu'
    ).to(device)
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss function with class weights if enabled
    if args.use_class_weights:
        from scripts.train_utils import get_class_weights
        print("\nCalculating positive class weights for imbalanced tasks...")
        pos_weight = get_class_weights(y_train, train_mask).to(device)
        print(f"  Class weights shape: {pos_weight.shape}")
        print(f"  Weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
        
        weights_path = output_dir / 'class_weights.json'
        with open(weights_path, 'w') as f:
            weights_list = pos_weight.cpu().tolist()
            task_weights = {task: weight for task, weight in zip(tasks, weights_list)}
            json.dump(task_weights, f, indent=2)
        print(f"  Saved class weights to {weights_path}")
        
        criterion = MaskedBCELoss(pos_weight=pos_weight, reduction='mean')
        print("  Using weighted BCE loss with positive class weights")
    else:
        criterion = MaskedBCELoss(reduction='mean')
        print("  Using standard BCE loss (no class weights)")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    print(f"\nTraining model for {args.n_epochs} epochs...")
    checkpoint_path = output_dir / 'best_model_checkpoint.pt'
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        threshold=args.threshold,
        checkpoint_path=str(checkpoint_path)
    )
    
    print("Saving validation probabilities...")
    _, _, val_probs = validate_epoch(model, val_loader, criterion, device, args.threshold)
    np.save(output_dir / 'val_probs.npy', val_probs)
    
    # Save final model
    model_path = output_dir / 'final_model.pt'
    model_save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': args.hidden_dims,
            'output_dim': output_dim,
            'dropout_rate': args.dropout_rate,
            'activation': 'relu'
        },
        'preprocessor_path': str(preprocessor_path),
        'history': history,
        'tuned_params_used': tuned_params is not None
    }
    
    if args.use_class_weights:
        model_save_dict['class_weights_used'] = True
        model_save_dict['class_weights_path'] = str(weights_path)
    
    torch.save(model_save_dict, model_path)
    print(f"\nSaved final model to {model_path}")
    
    # Plot training history
    if len(history['train_loss']) > 0:
        history_path = output_dir / 'training_history.png'
        plot_training_history(history, save_path=str(history_path))
    
    
    # Evaluate on test set (if not empty)
    print("\nEvaluating on test set...")
    if len(test_df) > 0 and X_test.shape[0] > 0:
        model.eval()

        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # FIX: Dùng X_test_scaled đã transform, không dùng X_test gốc
        with torch.no_grad():
            test_outputs = model(X_test_tensor)  # Dùng tensor đã scale
            test_probabilities = torch.sigmoid(test_outputs).cpu().numpy()
            test_predictions = (test_probabilities >= args.threshold).astype(int)
        
        # Save test probabilities
        np.save(output_dir / 'test_probs.npy', test_probabilities)
        print(f"Saved test probabilities to {output_dir / 'test_probs.npy'}")
        
        # Compute test metrics
        test_metrics = compute_metrics(y_test, test_predictions, 
                                    test_probabilities, test_mask, args.threshold)
        
        print("\nTest Set Metrics:")
        print(f"  Macro PR-AUC: {test_metrics['macro_pr_auc']:.4f}")
        print(f"  Macro ROC-AUC: {test_metrics['macro_roc_auc']:.4f}")
        print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Micro PR-AUC: {test_metrics['micro_pr_auc']:.4f}")
        print(f"  Micro ROC-AUC: {test_metrics['micro_roc_auc']:.4f}")
    else:
        print("⚠️ Test set is empty. Skipping test evaluation.")
        test_metrics = {
            'macro_pr_auc': 0.0,
            'macro_roc_auc': 0.0,
            'macro_f1': 0.0,
            'micro_pr_auc': 0.0,
            'micro_roc_auc': 0.0
        }
        # Save empty test probabilities
        np.save(output_dir / 'test_probs.npy', np.empty((0, len(tasks))))
    
    # Compare with baseline
    try:
        baseline_path = Path('models/baseline/xgboost_metrics.json')
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
            
            baseline_pr_auc = baseline_metrics['test']['macro_pr_auc']
            mlp_pr_auc = test_metrics['macro_pr_auc']
            
            print(f"\nComparison with Baseline (XGBoost):")
            print(f"  Baseline PR-AUC: {baseline_pr_auc:.4f}")
            print(f"  MLP PR-AUC: {mlp_pr_auc:.4f}")
            print(f"  Difference: {mlp_pr_auc - baseline_pr_auc:.4f}")
            
            if mlp_pr_auc > baseline_pr_auc:
                print("  ✅ MLP outperforms baseline!")
            else:
                print("  ⚠️  MLP does not outperform baseline.")
    except:
        print("\nCould not load baseline metrics for comparison.")
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'model_config': {
            'hidden_dims': args.hidden_dims,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'use_class_weights': args.use_class_weights
        },
        'training_history': {
            'best_epoch': history['best_epoch'],
            'best_val_metric': history['best_val_metric'],
            'best_val_loss': history['best_val_loss']
        },
        'tuned_params_used': tuned_params is not None
    }
    
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved test results to {results_path}")
    
    # Create per-task performance comparison
    task_performance = []
    for task_idx, task_name in enumerate(tasks):
        task_performance.append({
            'task': task_name,
            'mlp_pr_auc': test_metrics.get(f'task_{task_idx}_pr_auc', np.nan),
            'mlp_roc_auc': test_metrics.get(f'task_{task_idx}_roc_auc', np.nan),
        })
    
    # Add class weights to per-task performance if used
    if args.use_class_weights:
        weights_list = pos_weight.cpu().tolist()
        for idx, task_dict in enumerate(task_performance):
            task_dict['class_weight'] = weights_list[idx]
    
    task_df = pd.DataFrame(task_performance)
    task_path = output_dir / 'per_task_performance.csv'
    task_df.to_csv(task_path, index=False)
    print(f"Saved per-task performance to {task_path}")
    
    # Plot per-task performance
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tasks))
    width = 0.35
    
    ax.bar(x - width/2, task_df['mlp_pr_auc'], width, label='PR-AUC', color='skyblue')
    ax.bar(x + width/2, task_df['mlp_roc_auc'], width, label='ROC-AUC', color='lightcoral')
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Score')
    ax.set_title('MLP Performance per Task (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    task_plot_path = output_dir / 'per_task_performance.png'
    plt.savefig(task_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nDescriptor MLP training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Tuned parameters used: {tuned_params is not None}")
    print(f"Class weights used: {args.use_class_weights}")

if __name__ == '__main__':
    main()