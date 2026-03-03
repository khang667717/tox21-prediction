#!/usr/bin/env python3
"""
Interpretability analysis for Tox21 models.
Includes:
- SHAP analysis for tree-based models
- Gradient-based attribution for neural networks
- Feature importance analysis
"""
import torch.nn as nn

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Any, Optional
import warnings
import pickle

# Import our models
from scripts.baseline_model import featurize_ecfp, load_data
from scripts.descriptor_mlp import DescriptorCalculator, DescriptorPreprocessor, MLPModel
from scripts.gnn_model import MolecularGraphDataset, GNNModel, collate_molecular_graphs
from scripts.train_utils import compute_metrics, set_seed

warnings.filterwarnings('ignore')
set_seed(42)

class ModelInterpreter:
    def __init__(self, model_dir: str, data_dir: str, output_dir: str, model_type: str):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

        self.output_dir = Path(output_dir) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.load_data()
        
    def load_data(self):
        """Load processed data."""
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.val_df = pd.read_csv(self.data_dir / 'val.csv')
        self.test_df = pd.read_csv(self.data_dir / 'test.csv')
        
        with open(self.data_dir / 'tasks.json', 'r') as f:
            self.tasks = json.load(f)
        
    def analyze(self):
        """Run interpretability analysis."""
        raise NotImplementedError

class XGBoostInterpreter(ModelInterpreter):
    def __init__(self, model_dir: str, data_dir: str, output_dir: str, model_type: str):
        super().__init__(model_dir, data_dir, output_dir, model_type)

        
        # Load models
        self.models = []
        for task_idx in range(len(self.tasks)):
            model_path = self.model_dir / f'xgboost_task_{task_idx}.joblib'
            if model_path.exists():
                import joblib
                self.models.append(joblib.load(model_path))
            else:
                self.models.append(None)
        
        # Prepare features
        self.X_train = featurize_ecfp(self.train_df['canonical_smiles'].tolist())
        self.X_test = featurize_ecfp(self.test_df['canonical_smiles'].tolist())
    
    def shap_analysis(self, task_idx: int, n_samples: int = 100):
        """Perform SHAP analysis for a specific task."""
        if self.models[task_idx] is None:
            print(f"No model for task {task_idx}")
            return
        
        model = self.models[task_idx]
        
        # Sample data for SHAP (for speed)
        if n_samples < len(self.X_train):
            sample_indices = np.random.choice(len(self.X_train), n_samples, replace=False)
            X_sample = self.X_train[sample_indices]
        else:
            X_sample = self.X_train
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary Plot - Task {self.tasks[task_idx]}')
        plt.tight_layout()
        
        plot_path = self.output_dir / f'shap_task_{task_idx}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved SHAP plot to {plot_path}")
        
        # Also create a bar plot of mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(top_indices)), mean_abs_shap[top_indices])
        plt.xticks(range(len(top_indices)), top_indices, rotation=90)
        plt.xlabel('Feature Index')
        plt.ylabel('Mean |SHAP|')
        plt.title(f'Top 20 Important Features - Task {self.tasks[task_idx]}')
        plt.tight_layout()
        
        bar_path = self.output_dir / f'shap_bar_task_{task_idx}.png'
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
    
    def analyze(self, n_tasks: int = 3, n_samples: int = 100):
        """Run interpretability analysis for multiple tasks."""
        print("Running SHAP analysis for XGBoost models...")
        
        for task_idx in range(min(n_tasks, len(self.tasks))):
            if self.models[task_idx] is not None:
                print(f"\nAnalyzing task {task_idx}: {self.tasks[task_idx]}")
                self.shap_analysis(task_idx, n_samples)

class MLPInterpreter(ModelInterpreter):
    def __init__(self, model_dir: str, data_dir: str, output_dir: str, model_type: str):
        super().__init__(model_dir, data_dir, output_dir, model_type)

        
        # Load model
        model_path = self.model_dir / 'final_model.pt'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model_config = checkpoint['model_config']
            
            self.model = MLPModel(
                input_dim=self.model_config['input_dim'],
                hidden_dims=self.model_config['hidden_dims'],
                output_dim=self.model_config['output_dim'],
                dropout_rate=self.model_config['dropout_rate']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load preprocessor and descriptors
        preprocessor_path = self.model_dir / 'descriptor_preprocessor.pkl'
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        with open(self.model_dir / 'selected_descriptors.json', 'r') as f:
            self.descriptor_names = json.load(f)
        
        # Prepare features
        calculator = DescriptorCalculator()
        X_train_raw = calculator.compute_descriptors(self.train_df['canonical_smiles'].tolist())
        X_test_raw = calculator.compute_descriptors(self.test_df['canonical_smiles'].tolist())
        
        self.X_train = self.preprocessor.transform(X_train_raw)
        self.X_test = self.preprocessor.transform(X_test_raw)
    
    def gradient_analysis(self, task_idx: int, n_samples: int = 50):
        """Perform gradient-based attribution analysis."""
        # Convert to tensor
        X_tensor = torch.FloatTensor(self.X_test[:n_samples]).requires_grad_(True)
        
        # Forward pass
        output = self.model(X_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for specific task
        output[:, task_idx].sum().backward()
        
        # Get gradients
        gradients = X_tensor.grad.abs().mean(dim=0).detach().numpy()
        
        # Get top features
        top_indices = np.argsort(gradients)[-20:][::-1]
        top_gradients = gradients[top_indices]
        top_names = [self.descriptor_names[i] for i in top_indices]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_indices)), top_gradients)
        plt.xticks(range(len(top_indices)), top_names, rotation=90, fontsize=10)
        plt.xlabel('Descriptor')
        plt.ylabel('Mean Absolute Gradient')
        plt.title(f'Gradient-based Feature Importance - Task {self.tasks[task_idx]}')
        plt.tight_layout()
        
        plot_path = self.output_dir / f'gradient_task_{task_idx}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved gradient analysis to {plot_path}")
        
        return gradients
    
    def analyze(self, n_tasks: int = 3, n_samples: int = 50):
        """Run interpretability analysis."""
        print("Running gradient analysis for MLP model...")
        
        for task_idx in range(min(n_tasks, len(self.tasks))):
            print(f"\nAnalyzing task {task_idx}: {self.tasks[task_idx]}")
            self.gradient_analysis(task_idx, n_samples)

class GNNInterpreter(ModelInterpreter):
    def __init__(self, model_dir: str, data_dir: str, output_dir: str, model_type: str):
        super().__init__(model_dir, data_dir, output_dir, model_type)

        # Load model
        model_path = self.model_dir / 'final_model.pt'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model_config = checkpoint['model_config']
            
            self.model = GNNModel(
                node_feature_dim=self.model_config['node_feature_dim'],
                edge_feature_dim=self.model_config['edge_feature_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                num_layers=self.model_config['num_layers'],
                gnn_type=self.model_config['gnn_type'],
                num_heads=self.model_config['num_heads'],
                dropout_rate=self.model_config['dropout_rate'],
                output_dim=self.model_config['output_dim'],
                use_edge_features=self.model_config['use_edge_features'],
                pooling=self.model_config['pooling']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Create test dataset
        y_test = pd.read_csv(self.data_dir / 'test.csv')[self.tasks].values
        y_test = np.nan_to_num(y_test, nan=0.0)
        test_mask = np.load(self.data_dir / 'test_mask.npy')
        
        self.test_dataset = MolecularGraphDataset(
            self.test_df['canonical_smiles'].tolist(),
            y_test,
            test_mask
        )
    
    def node_importance_analysis(self, task_idx: int, molecule_idx: int = 0):
        data = self.test_dataset[molecule_idx]['graph']
        smiles = self.test_df.iloc[molecule_idx]['canonical_smiles']

        print(f"Analyzing molecule: {smiles}")
        print(f"Task: {self.tasks[task_idx]}")

        self.model.zero_grad()

        # Enable gradients on node features
        data.x = data.x.clone().detach().requires_grad_(True)

        out = self.model(data)
        out[0, task_idx].backward()

        # Gradient × Input
        node_importance = (data.x.grad * data.x).abs().sum(dim=1).detach().cpu().numpy()

        # Normalize
        if node_importance.max() > 0:
            node_importance /= node_importance.max()

        # ===== Visualization =====
        from rdkit import Chem
        from rdkit.Chem import Draw
        import matplotlib.pyplot as plt

        mol = Chem.MolFromSmiles(smiles)
        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(node_importance)), node_importance)
        plt.xticks(
            range(len(node_importance)),
            [f"{s}\n{i}" for i, s in enumerate(atom_symbols)],
            fontsize=9
        )
        plt.ylabel("Node Importance")
        plt.title(f"GNN Node Attribution – {self.tasks[task_idx]}")

        plot_path = self.output_dir / f"gnn_node_importance_task_{task_idx}_mol_{molecule_idx}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Saved: {plot_path}")

        
        # Create a wrapper for Captum
    
    def analyze(self, n_tasks: int = 2, n_molecules: int = 3):
        """Run interpretability analysis."""
        print("Running node importance analysis for GNN model...")
        
        for task_idx in range(min(n_tasks, len(self.tasks))):
            print(f"\nAnalyzing task {task_idx}: {self.tasks[task_idx]}")
            for mol_idx in range(min(n_molecules, len(self.test_dataset))):
                self.node_importance_analysis(task_idx, mol_idx)

def main():
    parser = argparse.ArgumentParser(description='Interpretability analysis for Tox21 models')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['xgboost', 'mlp', 'gnn'],
                       help='Type of model to interpret')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing the trained model')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--n_tasks', type=int, default=3,
                       help='Number of tasks to analyze')
    parser.add_argument('--output_dir', type=str, default='interpretability',
                       help='Directory to save interpretability results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize interpreter
    if args.model_type == 'xgboost':
        interpreter = XGBoostInterpreter(
            args.model_dir, args.data_dir, args.output_dir, 'xgboost'
        )
    elif args.model_type == 'mlp':
        interpreter = MLPInterpreter(
            args.model_dir, args.data_dir, args.output_dir, 'mlp'
        )
    elif args.model_type == 'gnn':
        interpreter = GNNInterpreter(
            args.model_dir, args.data_dir, args.output_dir, 'gnn'
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Run analysis
    interpreter.analyze(n_tasks=args.n_tasks)
    
    print(f"\nInterpretability analysis completed for {args.model_type} model")

if __name__ == '__main__':
    main()