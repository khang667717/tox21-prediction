#!/usr/bin/env python3
"""
Hyperparameter tuning for Tox21 models using Optuna.
FIXED: MLP return values and GNN categorical parameter.
"""

import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import xgboost as xgb
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
import sys

# Import our modules
from scripts.train_utils import MaskedBCELoss, compute_metrics, set_seed
from scripts.descriptor_mlp import DescriptorCalculator, DescriptorPreprocessor, MLPModel
from scripts.gnn_model import MolecularGraphDataset, GNNModel, collate_molecular_graphs

warnings.filterwarnings('ignore')
set_seed(42)


class HyperparameterTuner:
    """Base class for hyperparameter tuning."""
    
    def __init__(self, data_dir: str = 'data/processed', 
                 device: torch.device = None):
        self.data_dir = Path(data_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load processed data."""
        self.train_df = pd.read_csv(self.data_dir / 'train.csv')
        self.val_df = pd.read_csv(self.data_dir / 'val.csv')
        
        self.train_mask = np.load(self.data_dir / 'train_mask.npy')
        self.val_mask = np.load(self.data_dir / 'val_mask.npy')
        
        with open(self.data_dir / 'tasks.json', 'r') as f:
            self.tasks = json.load(f)
        
        # Prepare targets
        def prepare_targets(df, tasks):
            targets = df[tasks].values
            return np.nan_to_num(targets, nan=0.0)
        
        self.y_train = prepare_targets(self.train_df, self.tasks)
        self.y_val = prepare_targets(self.val_df, self.tasks)
        
    def prepare_features(self, model_type: str):
        """Prepare features for different model types."""
        if model_type == 'xgboost' or model_type == 'lightgbm':
            from scripts.baseline_model import featurize_ecfp
            X_train = featurize_ecfp(self.train_df['canonical_smiles'].tolist())
            X_val = featurize_ecfp(self.val_df['canonical_smiles'].tolist())
            return X_train, X_val
        
        elif model_type == 'mlp':
            calculator = DescriptorCalculator()
            X_train_raw = calculator.compute_descriptors(self.train_df['canonical_smiles'].tolist())
            X_val_raw = calculator.compute_descriptors(self.val_df['canonical_smiles'].tolist())
            
            preprocessor = DescriptorPreprocessor()
            preprocessor.fit(X_train_raw, calculator.get_descriptor_names())
            X_train = preprocessor.transform(X_train_raw)
            X_val = preprocessor.transform(X_val_raw)
            
            return X_train, X_val, preprocessor
        
        elif model_type == 'gnn':
            train_dataset = MolecularGraphDataset(
                self.train_df['canonical_smiles'].tolist(),
                self.y_train,
                self.train_mask
            )
            val_dataset = MolecularGraphDataset(
                self.val_df['canonical_smiles'].tolist(),
                self.y_val,
                self.val_mask
            )
            return train_dataset, val_dataset
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class XGBoostTuner(HyperparameterTuner):
    """Tuner for XGBoost model."""
    
    def __init__(self, data_dir: str = 'data/processed'):
        super().__init__(data_dir)
        self.model_type = 'xgboost'
        
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        
        X_train, X_val = self.prepare_features(self.model_type)
        
        n_tasks = len(self.tasks)
        task_scores = []
        
        for task_idx in range(n_tasks):
            task_mask = self.train_mask[:, task_idx].astype(bool)
            
            if task_mask.sum() > 0:
                X_task_train = X_train[task_mask]
                y_task_train = self.y_train[task_mask, task_idx]
                
                pos_ratio = y_task_train.mean()
                scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1
                params['scale_pos_weight'] = scale_pos_weight
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_task_train, y_task_train)
                
                task_mask_val = self.val_mask[:, task_idx].astype(bool)
                if task_mask_val.sum() > 0:
                    X_task_val = X_val[task_mask_val]
                    y_task_val = self.y_val[task_mask_val, task_idx]
                    
                    y_prob = model.predict_proba(X_task_val)[:, 1]
                    
                    from sklearn.metrics import average_precision_score
                    try:
                        pr_auc = average_precision_score(y_task_val, y_prob)
                        task_scores.append(pr_auc)
                    except:
                        task_scores.append(0.0)
        
        return np.mean(task_scores) if task_scores else 0.0


class MLPTuner(HyperparameterTuner):
    """Tuner for Descriptor MLP model."""
    
    def __init__(self, data_dir: str = 'data/processed'):
        super().__init__(data_dir)
        self.model_type = 'mlp'
        
    def create_data_loaders(self, X_train, y_train, mask_train, X_val, y_val, mask_val, batch_size):
        """Create data loaders with proper scaling."""
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, scaler
    
    def objective(self, trial: optuna.Trial) -> float:
        hidden_dims = []
        n_layers = trial.suggest_int('n_layers', 1, 3)
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f'hidden_dim_{i}', 64, 512))
        
        params = {
            'hidden_dims': hidden_dims,
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        }
        
        # Get features and preprocessor
        X_train, X_val, preprocessor = self.prepare_features(self.model_type)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(self.y_train)
        mask_train_tensor = torch.FloatTensor(self.train_mask)
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(self.y_val)
        mask_val_tensor = torch.FloatTensor(self.val_mask)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, mask_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, mask_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        input_dim = X_train.shape[1]
        output_dim = len(self.tasks)
        
        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            output_dim=output_dim,
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        criterion = MaskedBCELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        n_epochs = 30
        best_val_score = 0
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            for batch in train_loader:
                X_batch, y_batch, mask_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch, mask_batch)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            all_preds = []
            all_targets = []
            all_masks = []
            
            with torch.no_grad():
                for batch in val_loader:
                    X_batch, y_batch, mask_batch = batch
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device)
                    
                    outputs = model(X_batch)
                    probas = torch.sigmoid(outputs)
                    
                    all_preds.append(probas.cpu().numpy())
                    all_targets.append(y_batch.cpu().numpy())
                    all_masks.append(mask_batch.cpu().numpy())
            
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            all_masks = np.vstack(all_masks)
            
            metrics = compute_metrics(
                all_targets, 
                (all_preds > 0.5).astype(int), 
                all_preds, 
                all_masks
            )
            
            val_score = metrics['macro_pr_auc']
            
            if val_score > best_val_score:
                best_val_score = val_score
        
        return best_val_score

class GNNTuner(HyperparameterTuner):
    """Tuner for GNN model."""
    
    def __init__(self, data_dir: str = 'data/processed', gnn_subtype: str = None):
        super().__init__(data_dir)
        self.model_type = 'gnn'
        self.gnn_subtype = gnn_subtype
        if gnn_subtype is None:
            raise ValueError("gnn_subtype must be specified for GNN tuning")
    
    def objective(self, trial: optuna.Trial) -> float:
        # FIX: Không dùng suggest_categorical, dùng giá trị cố định
        gnn_type = self.gnn_subtype
        num_heads = 4 if gnn_type == 'gat' else 1  # GCN không dùng heads
        
        # Hidden dimension
        hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
        
        if gnn_type == 'gat':
            hidden_dim = (hidden_dim // num_heads) * num_heads
            if hidden_dim < 64:
                hidden_dim = 64
        
        params = {
            'hidden_dim': hidden_dim,
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'pooling': trial.suggest_categorical('pooling', ['mean', 'sum', 'max']),
            'gnn_type': gnn_type,
            'use_edge_features': True,
            'use_class_weights': True
        }
        
        # Chỉ thêm num_heads vào params nếu là GAT
        if gnn_type == 'gat':
            params['num_heads'] = num_heads
        
        train_dataset, val_dataset = self.prepare_features(self.model_type)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=collate_molecular_graphs,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_molecular_graphs,
            num_workers=0
        )
        
        # FIX: Thêm weight_decay parameter vào GNNModel
        model = GNNModel(
            node_feature_dim=train_dataset.get_node_feature_dim(),
            edge_feature_dim=train_dataset.get_edge_feature_dim(),
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            gnn_type=params['gnn_type'],
            num_heads=params.get('num_heads', 4),
            dropout_rate=params['dropout_rate'],
            output_dim=len(self.tasks),
            use_edge_features=params['use_edge_features'],
            pooling=params['pooling']
        ).to(self.device)
        
        criterion = MaskedBCELoss()
        optimizer = optim.AdamW(model.parameters(),
                               lr=params['learning_rate'],
                               weight_decay=params['weight_decay'])
        
        n_epochs = 20
        best_val_score = 0
        
        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                graph = batch['graph'].to(self.device)
                targets = batch['targets'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(graph)
                loss = criterion(outputs, targets, masks)
                loss.backward()
                optimizer.step()
            
            model.eval()
            all_preds = []
            all_targets = []
            all_masks = []
            
            with torch.no_grad():
                for batch in val_loader:
                    graph = batch['graph'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    
                    outputs = model(graph)
                    probas = torch.sigmoid(outputs)
                    
                    all_preds.append(probas.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_masks.append(masks.cpu().numpy())
            
            if all_preds:
                all_preds = np.vstack(all_preds)
                all_targets = np.vstack(all_targets)
                all_masks = np.vstack(all_masks)
                
                metrics = compute_metrics(all_targets,
                                        (all_preds > 0.5).astype(int),
                                        all_preds,
                                        all_masks)
                
                val_score = metrics['macro_pr_auc']
                
                if val_score > best_val_score:
                    best_val_score = val_score
        
        return best_val_score
    
def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for Tox21 models')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['xgboost', 'mlp', 'gnn'],
                       help='Type of model to tune')
    parser.add_argument('--gnn_subtype', type=str, default=None,
                       choices=['gcn', 'gat'],
                       help='Specific GNN type to tune (required if model_type=gnn)')
    parser.add_argument('--n_trials', type=int, default=30,
                       help='Number of Optuna trials')
    parser.add_argument('--output_dir', type=str, default='models/tuned',
                       help='Directory to save tuning results')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    args = parser.parse_args()
    
    # Validate GNN subtype
    if args.model_type == 'gnn' and args.gnn_subtype is None:
        parser.error("--gnn_subtype is required when --model_type=gnn")
    
    # Tạo output directory riêng cho từng subtype
    if args.model_type == 'gnn':
        output_subdir = f"{args.model_type}_{args.gnn_subtype}"
    else:
        output_subdir = args.model_type
    
    output_dir = Path(args.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tuner
    if args.model_type == 'xgboost':
        tuner = XGBoostTuner(args.data_dir)
    elif args.model_type == 'mlp':
        tuner = MLPTuner(args.data_dir)
    elif args.model_type == 'gnn':
        tuner = GNNTuner(args.data_dir, args.gnn_subtype)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Delete old study if exists
    storage_name = f'sqlite:///{output_dir}/optuna.db'
    try:
        optuna.delete_study(study_name=f'tox21_{output_subdir}_tuning', storage=storage_name)
        print(f"✅ Deleted existing study for {output_subdir}")
    except:
        pass
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'tox21_{output_subdir}_tuning',
        storage=storage_name,
        load_if_exists=False
    )
    
    study.optimize(tuner.objective, n_trials=args.n_trials, show_progress_bar=True)
    
    print(f"\nBest trial for {output_subdir}:")
    print(f"  Value (PR-AUC): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'trials': [
            {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in study.trials
        ]
    }
    
    results_path = output_dir / 'tuning_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    best_params_path = output_dir / 'best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    print(f"✅ Best parameters saved to {best_params_path}")

    # ========== THÊM ĐOẠN NÀY ĐỂ TẠO tuning_summary.json ==========
    # Tạo tuning_summary.json đơn giản cho pipeline đọc
    simple_params = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'model_type': args.model_type,
        'n_trials': args.n_trials,
        'trial_count': len(study.trials)
    }
    
    # Thêm gnn_subtype nếu là GNN
    if args.model_type == 'gnn' and args.gnn_subtype:
        simple_params['gnn_subtype'] = args.gnn_subtype
    
    summary_path = output_dir / 'tuning_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(simple_params, f, indent=2)
    
    print(f"✅ Tuning summary saved to {summary_path}")
    # ===============================================================

if __name__ == '__main__':
    main()