#!/usr/bin/env python3
"""
Graph Neural Network model for Tox21.
FIXED: Residual connections, LayerNorm, proper GAT edge features.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import LayerNorm  # FIX: Use LayerNorm instead of BatchNorm
from torch_geometric.nn import Set2Set

import numpy as np
import pandas as pd
from rdkit import Chem
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
from tqdm import tqdm

from scripts.train_utils import MaskedBCELoss, compute_metrics, set_seed
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
set_seed(42)

def plot_training_history(history: Dict[str, Any], save_path: str = None):
    """Plot training history for GNN."""
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
    
    # Plot metrics (PR-AUC)
    pr_aucs = [m.get('macro_pr_auc', 0) for m in history['val_metrics']]
    axes[1].plot(pr_aucs, label='PR-AUC', color='green')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PR-AUC')
    axes[1].set_title('Validation PR-AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved training history to {save_path}")
    
    plt.show()


class MolecularGraphDataset(torch.utils.data.Dataset):
    """Dataset for molecular graphs."""
    
    def __init__(self, smiles_list: List[str], targets: np.ndarray, 
                 masks: np.ndarray):
        self.smiles_list = smiles_list
        
        if len(smiles_list) == 0:
            self.targets = torch.empty((0, targets.shape[1])) if targets.ndim > 1 else torch.empty((0, 12))
            self.masks = torch.empty((0, masks.shape[1])) if masks.ndim > 1 else torch.empty((0, 12))
            self.graphs = []
            return
        
        if targets.dtype == np.object_:
            targets = targets.astype(np.float32)
        if masks.dtype == np.object_:
            masks = masks.astype(np.float32)
        
        self.targets = torch.FloatTensor(targets)
        self.masks = torch.FloatTensor(masks)
        
        print("Precomputing molecular graphs...")
        self.graphs = []
        
        for i, smiles in enumerate(tqdm(smiles_list)):
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
            else:
                # Dummy graph for invalid molecules
                dummy_graph = Data(
                    x=torch.zeros((1, self.get_node_feature_dim()), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, self.get_edge_feature_dim()), dtype=torch.float)
                )
                self.graphs.append(dummy_graph)
    
    @staticmethod
    def get_node_feature_dim() -> int:
        return 43
    
    @staticmethod
    def get_edge_feature_dim() -> int:
        return 4
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric graph."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
        
        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            
            # Atomic number (10 features)
            atomic_num = atom.GetAtomicNum()
            common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]
            for elem in common_atoms:
                features.append(1.0 if atomic_num == elem else 0.0)
            features.append(1.0 if atomic_num not in common_atoms else 0.0)
            
            # Degree (7 features)
            degree = atom.GetDegree()
            for d in range(6):
                features.append(1.0 if degree == d else 0.0)
            features.append(1.0 if degree > 5 else 0.0)
            
            # Formal charge (1)
            features.append(float(atom.GetFormalCharge()))
            
            # Radical electrons (1)
            features.append(float(atom.GetNumRadicalElectrons()))
            
            # Hybridization (6)
            hybridization = atom.GetHybridization()
            hybrid_types = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]
            for ht in hybrid_types:
                features.append(1.0 if hybridization == ht else 0.0)
            features.append(1.0 if hybridization not in hybrid_types else 0.0)
            
            # Aromatic (1)
            features.append(float(atom.GetIsAromatic()))
            
            # Ring info (7)
            features.append(float(atom.IsInRing()))
            features.append(float(atom.IsInRingSize(3)))
            features.append(float(atom.IsInRingSize(4)))
            features.append(float(atom.IsInRingSize(5)))
            features.append(float(atom.IsInRingSize(6)))
            features.append(float(atom.IsInRingSize(7)))
            features.append(float(atom.IsInRingSize(8)))
            
            # Hydrogen count (6)
            h_count = atom.GetTotalNumHs()
            for h in range(5):
                features.append(1.0 if h_count == h else 0.0)
            features.append(1.0 if h_count > 4 else 0.0)
            
            # Chirality (1)
            features.append(float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED))
            
            # Mass (1)
            features.append(float(atom.GetMass() / 100.0))
            
            # Implicit Hs (1)
            features.append(float(atom.GetNumImplicitHs()))
            
            # Valence (1)
            features.append(float(atom.GetTotalValence()))
            
            atom_features.append(features)
        
        # Edge features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            bond_type = bond.GetBondType()
            bond_features = [
                1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
                1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
                1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
                1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
            ]
            
            edge_features.append(bond_features)
            edge_features.append(bond_features)
        
        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]
            edge_features = [[0.0, 0.0, 0.0, 1.0]]
        
        x = torch.FloatTensor(atom_features)
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'graph': self.graphs[idx],
            'targets': self.targets[idx],
            'masks': self.masks[idx],
            'idx': idx
        }


def collate_molecular_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for molecular graph dataset."""
    graphs = [item['graph'] for item in batch]
    targets = torch.stack([item['targets'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])
    
    batched_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batched_graph,
        'targets': targets,
        'masks': masks,
        'indices': indices
    }


class GNNModel(nn.Module):
    """Graph Neural Network with residual connections and layer normalization."""
    
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 gnn_type: str = 'gcn',
                 num_heads: int = 4,
                 dropout_rate: float = 0.2,
                 output_dim: int = 12,
                 use_edge_features: bool = True,
                 pooling: str = 'mean',
                 residual: bool = True):
        super().__init__()
        
        self.gnn_type = gnn_type.lower()
        self.use_edge_features = use_edge_features
        self.pooling = pooling
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.residual = residual
        
        # Adjust hidden_dim for GAT
        if self.gnn_type == 'gat':
            self.hidden_dim = (hidden_dim // num_heads) * num_heads
            if self.hidden_dim != hidden_dim:
                print(f"  ⚠️ Adjusting hidden_dim from {hidden_dim} to {self.hidden_dim} for GAT")
        else:
            self.hidden_dim = hidden_dim
        
        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, self.hidden_dim),
            LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Edge embedding
        if use_edge_features and edge_feature_dim > 0:
            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_feature_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            self.edge_embedding = None
        
        # GNN layers with LayerNorm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if self.gnn_type == 'gcn':
                conv = GCNConv(self.hidden_dim, self.hidden_dim)
                
            elif self.gnn_type == 'gat':
                conv = GATConv(
                    self.hidden_dim,
                    self.hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout_rate,
                    concat=True
                )
                
            elif self.gnn_type == 'gine':
                mlp = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                )
                conv = GINEConv(mlp, edge_dim=self.hidden_dim if use_edge_features else None)
                
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            self.convs.append(conv)
            self.norms.append(LayerNorm(self.hidden_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim // 2, output_dim)
        )
    
    def forward(self, graph) -> torch.Tensor:
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        
        # Node embedding
        x = self.node_embedding(x)
        
        # Edge embedding
        edge_attr = None
        if self.use_edge_features and self.edge_embedding is not None:
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_attr = self.edge_embedding(graph.edge_attr)
        
        # For GAT, incorporate edge features by adding to node features
        if self.gnn_type == 'gat' and edge_attr is not None:
            # Add edge features to target nodes
            row, col = edge_index
            x = x.clone()
            x[col] = x[col] + edge_attr.mean(dim=1, keepdim=True)
        
        # GNN layers with residual connections
        for i in range(self.num_layers):
            x_prev = x  # Save for residual
            
            # Message passing
            if self.gnn_type == 'gine' and edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            
            # Normalization and activation
            x = self.norms[i](x)
            x = F.relu(x)
            
            # Residual connection (only if shapes match)
            if self.residual and x.shape == x_prev.shape:
                x = x + x_prev
            
            # Dropout
            x = self.dropout(x)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        
        # Readout
        x = self.readout(x)
        
        return x


def train_gnn_epoch(model: nn.Module, train_loader: DataLoader, 
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device, clip_grad: float = 1.0) -> float:
    """Train GNN model for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        graph = batch['graph'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)
        
        optimizer.zero_grad()
        outputs = model(graph)
        loss = criterion(outputs, targets, masks)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0


def validate_gnn(model: nn.Module, val_loader: DataLoader, 
                 criterion: nn.Module, device: torch.device,
                 threshold: float = 0.5) -> Tuple[float, Dict[str, Any], np.ndarray]:
    """Validate GNN model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    all_masks = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            graph = batch['graph'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            outputs = model(graph)
            loss = criterion(outputs, targets, masks)
            
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
    
    if all_predictions:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_masks = np.vstack(all_masks)
        all_probabilities = np.vstack(all_probabilities)
        
        metrics = compute_metrics(all_targets, all_predictions, 
                                all_probabilities, all_masks, threshold)
    else:
        metrics = {}
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    
    return avg_loss, metrics, all_probabilities

def train_gnn_model(model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.ReduceLROnPlateau,
                    device: torch.device,
                    n_epochs: int = 100, patience: int = 15,
                    threshold: float = 0.5, 
                    checkpoint_path: str = None) -> Dict[str, Any]:
    """Train GNN model with early stopping."""
    
    best_val_loss = float('inf')
    best_val_metric = 0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Train
        train_loss = train_gnn_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics, _ = validate_gnn(model, val_loader, criterion, device, threshold)
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step(val_metrics.get('macro_pr_auc', 0))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        # Check for improvement
        val_metric = val_metrics.get('macro_pr_auc', 0)
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_metric': best_val_metric,
                    'val_loss': best_val_loss,
                }, checkpoint_path)
            
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val PR-AUC: {val_metric:.4f}")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
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


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for Tox21')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='models/gnn',
                       help='Directory to save model and results')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'gine'],
                       help='Type of GNN')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads (for GAT)')
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
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'sum', 'max'],
                       help='Global pooling method')
    parser.add_argument('--use_edge_features', action='store_true',
                       help='Use edge features')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use positive class weights')
    parser.add_argument('--weight_clip', type=float, default=15.0,
                       help='Maximum value for class weights clipping')
    parser.add_argument('--residual', action='store_true', default=True,
                       help='Use residual connections')
    parser.add_argument('--tuned_params', type=str, default=None,
                       help='Path to tuned parameters JSON file')
    args = parser.parse_args()
    
    # Load tuned parameters
    tuned_params = None
    if args.tuned_params:
        try:
            # FIX: Resolve path properly
            tuned_params_path = Path(args.tuned_params)
            
            # Try multiple possible paths
            if not tuned_params_path.exists():
                tuned_params_path = Path.cwd() / args.tuned_params
            if not tuned_params_path.exists():
                tuned_params_path = Path(__file__).parent.parent / args.tuned_params
            if not tuned_params_path.exists():
                tuned_params_path = Path(__file__).parent / args.tuned_params
                
            print(f"\n  📍 Looking for tuned params at: {tuned_params_path}")
            
            if not tuned_params_path.exists():
                print(f"  ❌ Tuned params file not found at any location!")
                print(f"  ⚠️  Using default parameters")
            else:
                with open(tuned_params_path, 'r') as f:
                    tuning_results = json.load(f)
                    
                # Handle both formats
                if 'best_params' in tuning_results:
                    tuned_params = tuning_results['best_params']
                    print(f"  📦 Loaded legacy format (wrapped in 'best_params')")
                else:
                    tuned_params = tuning_results
                    print(f"  📦 Loaded direct parameter format")
                
                print(f"  ✅ Successfully loaded tuned parameters")
                print(f"\n  📊 TUNED PARAMETERS:")
                
                # Override hyperparameters
                param_mapping = {
                    'hidden_dim': 'hidden_dim',
                    'num_layers': 'num_layers',
                    'num_heads': 'num_heads',
                    'dropout_rate': 'dropout_rate',
                    'learning_rate': 'learning_rate',
                    'batch_size': 'batch_size',
                    'weight_decay': 'weight_decay',
                    'pooling': 'pooling',
                    'gnn_type': 'gnn_type',
                    'use_edge_features': 'use_edge_features',
                    'use_class_weights': 'use_class_weights'
                }
                
                for tuned_key, arg_key in param_mapping.items():
                    if tuned_key in tuned_params:
                        old_value = getattr(args, arg_key)
                        new_value = tuned_params[tuned_key]
                        setattr(args, arg_key, new_value)
                        print(f"    ✅ {arg_key:20} : {old_value} -> {new_value}")
                    else:
                        print(f"    ⚠️  {arg_key:20} : NOT FOUND in tuned params")
                        
        except Exception as e:
            print(f"\n  ❌ Could not load tuned parameters: {e}")
            print(f"  ⚠️  Using default parameters")
    else:
        print(f"\n  ℹ️  No tuned parameters provided, using defaults")
        
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
    test_df = pd.read_csv(data_dir / 'test.csv') if (data_dir / 'test.csv').exists() else pd.DataFrame()
    
    train_mask = np.load(data_dir / 'train_mask.npy')
    val_mask = np.load(data_dir / 'val_mask.npy')
    test_mask = np.load(data_dir / 'test_mask.npy') if (data_dir / 'test_mask.npy').exists() else np.empty((0, 12))
    
    with open(data_dir / 'tasks.json', 'r') as f:
        tasks = json.load(f)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare targets
    def prepare_targets(df, tasks):
        if len(df) == 0:
            return np.empty((0, len(tasks)), dtype=np.float32)
        targets = df[tasks].values
        return np.nan_to_num(targets, nan=0.0).astype(np.float32)
    
    y_train = prepare_targets(train_df, tasks)
    y_val = prepare_targets(val_df, tasks)
    y_test = prepare_targets(test_df, tasks)
    
    # Create datasets
    print("\nCreating molecular graph datasets...")
    train_dataset = MolecularGraphDataset(train_df['canonical_smiles'].tolist(), y_train, train_mask)
    val_dataset = MolecularGraphDataset(val_df['canonical_smiles'].tolist(), y_val, val_mask)
    test_dataset = MolecularGraphDataset(test_df['canonical_smiles'].tolist(), y_test, test_mask) if len(test_df) > 0 else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                            collate_fn=collate_molecular_graphs, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                          collate_fn=collate_molecular_graphs, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_molecular_graphs, num_workers=0) if test_dataset is not None else None
    
    # Initialize model
    node_feature_dim = MolecularGraphDataset.get_node_feature_dim()
    edge_feature_dim = MolecularGraphDataset.get_edge_feature_dim()
    output_dim = len(tasks)
    
    print(f"\nInitializing GNN model ({args.gnn_type.upper()})...")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Number of heads: {args.num_heads}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Use edge features: {args.use_edge_features}")
    print(f"  Use class weights: {args.use_class_weights}")
    print(f"  Residual connections: {args.residual}")
    
    model = GNNModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        output_dim=output_dim,
        use_edge_features=args.use_edge_features,
        pooling=args.pooling,
        residual=args.residual
    ).to(device)
    
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if args.use_class_weights:
        from scripts.train_utils import get_class_weights
        print("\nCalculating positive class weights...")
        pos_weight = get_class_weights(y_train, train_mask, clip_max=args.weight_clip).to(device)
        print(f"  Weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
        
        weights_path = output_dir / 'class_weights.json'
        with open(weights_path, 'w') as f:
            json.dump({task: pos_weight[i].item() for i, task in enumerate(tasks)}, f, indent=2)
        
        criterion = MaskedBCELoss(pos_weight=pos_weight, reduction='mean')
        print("  Using weighted BCE loss")
    else:
        criterion = MaskedBCELoss(reduction='mean')
        print("  Using standard BCE loss")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Train model
    print(f"\nTraining model for {args.n_epochs} epochs...")
    checkpoint_path = output_dir / 'best_model_checkpoint.pt'
    
    history = train_gnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        threshold=args.threshold,
        checkpoint_path=str(checkpoint_path)
    )
    
    # Save validation probabilities
    print("Saving validation probabilities...")
    _, _, val_probs = validate_gnn(model, val_loader, criterion, device, args.threshold)
    np.save(output_dir / 'val_probs.npy', val_probs)
    
    # Save final model
    model_path = output_dir / 'final_model.pt'
    model_save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'gnn_type': args.gnn_type,
            'num_heads': args.num_heads,
            'dropout_rate': args.dropout_rate,
            'output_dim': output_dim,
            'use_edge_features': args.use_edge_features,
            'pooling': args.pooling,
            'residual': args.residual
        },
        'history': history,
        'tuned_params_used': tuned_params is not None
    }
    
    if args.use_class_weights:
        model_save_dict['class_weights_used'] = True
    
    torch.save(model_save_dict, model_path)
    print(f"\nSaved final model to {model_path}")

    history_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=str(history_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    if test_loader is not None:
        test_loss, test_metrics, test_probs = validate_gnn(
            model, test_loader, criterion, device, args.threshold
        )
        np.save(output_dir / 'test_probs.npy', test_probs)
        
        print("\nTest Set Metrics:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Macro PR-AUC: {test_metrics.get('macro_pr_auc', 0):.4f}")
        print(f"  Macro ROC-AUC: {test_metrics.get('macro_roc_auc', 0):.4f}")
        print(f"  Micro PR-AUC: {test_metrics.get('micro_pr_auc', 0):.4f}")
        print(f"  Micro ROC-AUC: {test_metrics.get('micro_roc_auc', 0):.4f}")
    else:
        print("⚠️ Test set is empty. Skipping test evaluation.")
        test_metrics = {'macro_pr_auc': 0.0, 'macro_roc_auc': 0.0}
        np.save(output_dir / 'test_probs.npy', np.empty((0, len(tasks))))
    
    # Save test results
    results = {
        'test_metrics': test_metrics,
        'test_loss': test_loss if test_loader else 0.0,
        'model_config': {
            'gnn_type': args.gnn_type,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'pooling': args.pooling,
            'use_edge_features': args.use_edge_features,
            'use_class_weights': args.use_class_weights,
            'residual': args.residual
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
    print(f"\nGNN training completed!")
    print(f"Results saved to: {output_dir}")
    
    return test_metrics.get('macro_pr_auc', 0)


if __name__ == '__main__':
    main()