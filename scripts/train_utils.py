
#!/usr/bin/env python3
"""
Training utilities for Tox21 project.
Includes masked loss calculation with positive weighting, metric computation, and helper functions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    f1_score, precision_score, recall_score,
    accuracy_score
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

class MaskedBCELoss(nn.Module):
    """Masked Binary Cross Entropy with optional positive weighting for multi-task learning."""
    def __init__(self, pos_weight=None, reduction='mean'):
        """
        Args:
            pos_weight: Tensor of shape (n_tasks,) with weight for positive samples
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, n_tasks) logits
            targets: (batch_size, n_tasks) floats 0/1
            mask: (batch_size, n_tasks), 1 for valid, 0 for invalid
        """
        # Calculate loss per element with optional positive weighting
        loss = F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Apply mask
        loss = loss * mask
        
        # Normalize by number of valid elements
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    y_prob: np.ndarray,
                    mask: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute evaluation metrics for multi-task classification.
    
    Args:
        y_true: (n_samples, n_tasks) ground truth
        y_pred: (n_samples, n_tasks) predicted labels (binary)
        y_prob: (n_samples, n_tasks) predicted probabilities
        mask: (n_samples, n_tasks) mask for valid labels
        threshold: threshold for binary classification
    
    Returns:
        Dictionary containing metrics
    """
    metrics = {}
    n_tasks = y_true.shape[1]
    
    # Initialize arrays for per-task metrics
    task_roc_auc = np.zeros(n_tasks)
    task_pr_auc = np.zeros(n_tasks)
    task_f1 = np.zeros(n_tasks)
    task_precision = np.zeros(n_tasks)
    task_recall = np.zeros(n_tasks)
    task_accuracy = np.zeros(n_tasks)
    
    for task_idx in range(n_tasks):
        # Get mask for current task
        task_mask = mask[:, task_idx].astype(bool)
        
        if task_mask.sum() > 0:  # Only compute if we have samples
            y_true_task = y_true[task_mask, task_idx]
            y_prob_task = y_prob[task_mask, task_idx]
            y_pred_task = y_pred[task_mask, task_idx]
            
            try:
                task_roc_auc[task_idx] = roc_auc_score(y_true_task, y_prob_task)
            except:
                task_roc_auc[task_idx] = np.nan
            
            try:
                task_pr_auc[task_idx] = average_precision_score(y_true_task, y_prob_task)
            except:
                task_pr_auc[task_idx] = np.nan
            
            try:
                task_f1[task_idx] = f1_score(y_true_task, y_pred_task)
            except:
                task_f1[task_idx] = np.nan
                
            try:
                task_precision[task_idx] = precision_score(y_true_task, y_pred_task)
            except:
                task_precision[task_idx] = np.nan
                
            try:
                task_recall[task_idx] = recall_score(y_true_task, y_pred_task)
            except:
                task_recall[task_idx] = np.nan
                
            try:
                task_accuracy[task_idx] = accuracy_score(y_true_task, y_pred_task)
            except:
                task_accuracy[task_idx] = np.nan
        else:
            task_roc_auc[task_idx] = np.nan
            task_pr_auc[task_idx] = np.nan
            task_f1[task_idx] = np.nan
            task_precision[task_idx] = np.nan
            task_recall[task_idx] = np.nan
            task_accuracy[task_idx] = np.nan
    
    # Compute macro averages (ignore nan)
    metrics['macro_roc_auc'] = np.nanmean(task_roc_auc)
    metrics['macro_pr_auc'] = np.nanmean(task_pr_auc)
    metrics['macro_f1'] = np.nanmean(task_f1)
    metrics['macro_precision'] = np.nanmean(task_precision)
    metrics['macro_recall'] = np.nanmean(task_recall)
    metrics['macro_accuracy'] = np.nanmean(task_accuracy)
    
    # Compute micro averages (pool all tasks)
    y_true_flat = y_true[mask.astype(bool)]
    y_pred_flat = y_pred[mask.astype(bool)]
    y_prob_flat = y_prob[mask.astype(bool)]
    
    if len(y_true_flat) > 0:
        try:
            metrics['micro_roc_auc'] = roc_auc_score(y_true_flat, y_prob_flat)
        except:
            metrics['micro_roc_auc'] = np.nan
            
        try:
            metrics['micro_pr_auc'] = average_precision_score(y_true_flat, y_prob_flat)
        except:
            metrics['micro_pr_auc'] = np.nan
            
        try:
            metrics['micro_f1'] = f1_score(y_true_flat, y_pred_flat)
        except:
            metrics['micro_f1'] = np.nan
            
        try:
            metrics['micro_precision'] = precision_score(y_true_flat, y_pred_flat)
        except:
            metrics['micro_precision'] = np.nan
            
        try:
            metrics['micro_recall'] = recall_score(y_true_flat, y_pred_flat)
        except:
            metrics['micro_recall'] = np.nan
            
        try:
            metrics['micro_accuracy'] = accuracy_score(y_true_flat, y_pred_flat)
        except:
            metrics['micro_accuracy'] = np.nan
    else:
        metrics['micro_roc_auc'] = np.nan
        metrics['micro_pr_auc'] = np.nan
        metrics['micro_f1'] = np.nan
        metrics['micro_precision'] = np.nan
        metrics['micro_recall'] = np.nan
        metrics['micro_accuracy'] = np.nan
    
    # Store per-task metrics
    for task_idx in range(n_tasks):
        metrics[f'task_{task_idx}_roc_auc'] = task_roc_auc[task_idx]
        metrics[f'task_{task_idx}_pr_auc'] = task_pr_auc[task_idx]
        metrics[f'task_{task_idx}_f1'] = task_f1[task_idx]
        metrics[f'task_{task_idx}_precision'] = task_precision[task_idx]
        metrics[f'task_{task_idx}_recall'] = task_recall[task_idx]
        metrics[f'task_{task_idx}_accuracy'] = task_accuracy[task_idx]
    
    # Add sample counts
    metrics['n_samples'] = mask.shape[0]
    metrics['n_valid_labels'] = int(mask.sum())
    
    return metrics

def get_class_weights(y_train: np.ndarray, mask_train: np.ndarray, 
                      clip_max: float = 15.0) -> torch.Tensor:  # Đổi từ 50.0 -> 15.0
    """
    Calculate positive class weights for imbalanced multi-task learning.
    Args:
        y_train: (n_samples, n_tasks) training labels
        mask_train: (n_samples, n_tasks) mask for valid labels
        clip_max: Maximum weight value (default: 15.0)  # Đổi từ 50.0 -> 15.0
    """
    n_tasks = y_train.shape[1]
    pos_weights = torch.ones(n_tasks)
    
    for task_idx in range(n_tasks):
        task_mask = mask_train[:, task_idx].astype(bool)
        if task_mask.sum() > 0:
            y_task = y_train[task_mask, task_idx]
            num_pos = y_task.sum()
            num_neg = len(y_task) - num_pos
            
            if num_pos > 0:
                weight = num_neg / num_pos
                # Clip extreme weights
                if weight > clip_max:
                    print(f"  ⚠️ Task {task_idx}: weight={weight:.2f} > {clip_max}, clipping to {clip_max}")
                    weight = clip_max
                pos_weights[task_idx] = weight
    
    return pos_weights
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_proba_with_masks(model, X, mask=None, device='cpu'):
    """
    Make predictions with proper masking.
    
    Args:
        model: Trained model (sklearn, xgboost, or torch)
        X: Features
        mask: Optional mask to apply to predictions
        device: Device for torch models
    """
    # For tree-based models (sklearn, xgboost)
    if hasattr(model, 'predict_proba'):
        if hasattr(model, 'classes_'):  # Single output classifier
            try:
                proba = model.predict_proba(X)[:, 1]
            except:
                proba = model.predict_proba(X)
        else:  # Multi-output classifier
            proba = model.predict_proba(X)
            if isinstance(proba, list) or isinstance(proba, tuple):
                # Handle case where predict_proba returns list of arrays
                proba = np.array([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in proba]).T
            else:
                proba = np.array([p[:, 1] for p in proba]).T
    # For PyTorch models
    elif hasattr(model, 'forward'):
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(device)
            else:
                X_tensor = X.to(device)
            
            logits = model(X_tensor)
            proba = torch.sigmoid(logits).cpu().numpy()
    else:
        # Fallback for other model types
        proba = model.predict(X)
    
    # Ensure proba is 2D
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    
    # Apply mask if provided
    if mask is not None:
        proba = proba * mask
    
    return proba