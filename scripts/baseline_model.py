
#!/usr/bin/env python3
"""
Baseline model: ECFP4 fingerprints + XGBoost for Tox21 dataset.
Now with optional Spark featurization and distributed MLlib baseline.
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import dump, load
import json
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time

from scripts.train_utils import compute_metrics, set_seed

# -------------------- Spark optional import --------------------
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, IntegerType, DoubleType
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
# -------------------------------------------------------------

warnings.filterwarnings('ignore')

# Set random seed
set_seed(42)

def featurize_ecfp(smiles_list: List[str], radius: int = 2, n_bits: int = 2048, 
                   use_spark: bool = False, spark_session=None) -> np.ndarray:
    """
    Convert SMILES to ECFP fingerprints.
    FIXED: Handle empty SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        radius: ECFP radius
        n_bits: Number of bits in fingerprint
        use_spark: Use Spark UDF for parallelization
        spark_session: Active Spark session
    
    Returns:
        numpy array of shape (n_samples, n_bits)
    """
    # FIX: Handle empty list
    if len(smiles_list) == 0:
        return np.empty((0, n_bits), dtype=np.int32)
    
    if use_spark and SPARK_AVAILABLE and spark_session is not None:
        print("  🔥 Spark featurization enabled")
        
        def fp_as_array(smiles):
            """Convert SMILES to fingerprint array."""
            if not smiles:
                return [0] * n_bits
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [0] * n_bits
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            return [int(b) for b in fp]
        
        fp_udf = udf(fp_as_array, ArrayType(IntegerType()))
        
        # Create DataFrame with SMILES
        df_spark = spark_session.createDataFrame([(s,) for s in smiles_list], ["smiles"])
        
        # Apply UDF
        df_spark = df_spark.withColumn("fp", fp_udf(col("smiles")))
        
        # Collect to driver
        fingerprints = np.array(df_spark.select("fp").rdd.map(lambda r: r[0]).collect(), dtype=np.int32)
        
        return fingerprints
    else:
        # Original pandas implementation
        fingerprints = []
        invalid_count = 0
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_count += 1
                fingerprints.append(np.zeros(n_bits, dtype=int))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fingerprints.append(np.array(fp))
        
        print(f"Featurized {len(smiles_list)} molecules, {invalid_count} invalid.")
        
        return np.array(fingerprints)

def train_spark_baseline(train_df, train_mask, tasks, spark, radius=2, n_bits=2048):
    """
    Train a simple LogisticRegression with MLlib as Big Data demonstration.
    Not meant to replace XGBoost, just to show distributed capability.
    """
    print("\n🔥 Spark MLlib baseline (Logistic Regression) - distributed training")
    
    # Prepare data: only use tasks with enough samples
    valid_tasks = []
    for i, task in enumerate(tasks):
        mask_col = train_mask[:, i].astype(bool)
        if mask_col.sum() > 50:
            valid_tasks.append((i, task))
    
    # Limit to 3 tasks for demo
    demo_tasks = valid_tasks[:3]
    print(f"  Training Spark LR on {len(demo_tasks)} tasks: {[t[1] for t in demo_tasks]}")
    
    results = {}
    for task_idx, task_name in demo_tasks:
        print(f"  Training Spark LR for {task_name}...")
        
        # Get masked data
        mask = train_mask[:, task_idx].astype(bool)
        X = featurize_ecfp(train_df.iloc[mask]['canonical_smiles'].tolist(),
                          radius=radius, n_bits=n_bits,
                          use_spark=True, spark_session=spark)
        y = train_df.iloc[mask][task_name].values.astype(float)
        
        # Create Spark DataFrame
        feature_cols = [f"f{i}" for i in range(X.shape[1])]
        pdf = pd.DataFrame(X, columns=feature_cols)
        pdf['label'] = y
        sdf = spark.createDataFrame(pdf)
        
        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        sdf = assembler.transform(sdf).select("features", "label")
        
        # Train logistic regression
        lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.0)
        model = lr.fit(sdf)
        
        # Quick evaluation on training (just for demo)
        evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
        pred = model.transform(sdf)
        pr_auc = evaluator.evaluate(pred)
        results[task_name] = pr_auc
        print(f"     Spark LR PR-AUC (train): {pr_auc:.4f}")
    
    print(f"  ✅ Spark MLlib baseline completed. Results: {results}")
    return results

def load_data(data_dir: str = 'data/processed'):
    """Load processed data and masks."""
    data_dir = Path(data_dir)
    
    # Load splits
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    # Load masks
    train_mask = np.load(data_dir / 'train_mask.npy')
    val_mask = np.load(data_dir / 'val_mask.npy')
    test_mask = np.load(data_dir / 'test_mask.npy')
    
    # Load tasks
    with open(data_dir / 'tasks.json', 'r') as f:
        tasks = json.load(f)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Tasks: {tasks}")
    
    return train_df, val_df, test_df, train_mask, val_mask, test_mask, tasks

def prepare_targets(df: pd.DataFrame, tasks: List[str]) -> np.ndarray:
    """Extract target values from dataframe."""
    targets = df[tasks].values
    
    # Replace NaN with 0 (will be masked during training)
    targets = np.nan_to_num(targets, nan=0.0)
    
    return targets

def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray, 
                        mask_train: np.ndarray, tasks: List[str],
                        tuned_params: Dict[str, Any] = None,
                        params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Train XGBoost model for multi-task classification.
    
    Returns:
        Dictionary containing trained models and results
    """
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
    
    # Override with tuned parameters if available
    if tuned_params:
        for key in ['max_depth', 'learning_rate', 'n_estimators', 
                   'subsample', 'colsample_bytree', 'min_child_weight',
                   'gamma', 'reg_alpha', 'reg_lambda']:
            if key in tuned_params:
                params[key] = tuned_params[key]
        print(f"Using tuned parameters: {params}")
    
    n_tasks = len(tasks)
    models = []
    
    print("Training XGBoost models for each task...")
    
    for task_idx in range(n_tasks):
        task_mask = mask_train[:, task_idx].astype(bool)
        
        if task_mask.sum() > 0:
            X_task = X_train[task_mask]
            y_task = y_train[task_mask, task_idx]
            
            # Calculate class weights for imbalance
            pos_ratio = y_task.mean()
            scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1
            
            task_params = params.copy()
            task_params['scale_pos_weight'] = scale_pos_weight
            
            model = xgb.XGBClassifier(**task_params)
            model.fit(X_task, y_task)
            models.append(model)
        else:
            models.append(None)
        
        if (task_idx + 1) % 3 == 0 or task_idx == n_tasks - 1:
            print(f"  Trained {task_idx + 1}/{n_tasks} tasks")
    
    return {'models': models, 'params': params}

def evaluate_model(models: List[Any], X: np.ndarray, y_true: np.ndarray,
                   mask: np.ndarray, tasks: List[str], 
                   threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate trained models on given data.
    FIXED: Return y_prob for saving test predictions.
    
    Returns:
        Dictionary containing predictions and metrics
    """
    n_tasks = len(tasks)
    n_samples = X.shape[0]
    
    y_pred = np.zeros((n_samples, n_tasks))
    y_prob = np.zeros((n_samples, n_tasks))
    
    print("Making predictions for each task...")
    
    for task_idx, model in enumerate(models):
        if model is not None:
            try:
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if len(proba.shape) == 2 and proba.shape[1] == 2:
                        y_prob[:, task_idx] = proba[:, 1]
                    else:
                        y_prob[:, task_idx] = proba
                else:
                    y_prob[:, task_idx] = model.predict(X)
                
                # Apply threshold
                y_pred[:, task_idx] = (y_prob[:, task_idx] >= threshold).astype(int)
            except Exception as e:
                print(f"Error predicting task {task_idx}: {e}")
                y_pred[:, task_idx] = 0
                y_prob[:, task_idx] = 0
        else:
            y_pred[:, task_idx] = 0
            y_prob[:, task_idx] = 0
    
    # Apply mask to predictions
    y_pred = y_pred * mask
    y_prob = y_prob * mask
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, mask, threshold)
    
    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'metrics': metrics
    }

def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, 
                    mask: np.ndarray, tasks: List[str],
                    save_path: str = None):
    """Plot ROC curves for each task."""
    n_tasks = len(tasks)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for task_idx in range(n_tasks):
        ax = axes[task_idx]
        task_mask = mask[:, task_idx].astype(bool)
        
        if task_mask.sum() > 0:
            y_true_task = y_true[task_mask, task_idx]
            y_prob_task = y_prob[task_mask, task_idx]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_task, y_prob_task)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{tasks[task_idx]}')
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{tasks[task_idx]}')
    
    # Hide unused subplots
    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('ROC Curves for Each Task', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.show()

def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray,
                   mask: np.ndarray, tasks: List[str],
                   save_path: str = None):
    """Plot Precision-Recall curves for each task."""
    from sklearn.metrics import precision_recall_curve, auc
    
    n_tasks = len(tasks)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for task_idx in range(n_tasks):
        ax = axes[task_idx]
        task_mask = mask[:, task_idx].astype(bool)
        
        if task_mask.sum() > 0:
            y_true_task = y_true[task_mask, task_idx]
            y_prob_task = y_prob[task_mask, task_idx]
            
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(y_true_task, y_prob_task)
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, color='darkorange', lw=2,
                   label=f'PR (AUC = {pr_auc:.2f})')
            
            # Plot baseline (fraction of positives)
            pos_ratio = y_true_task.mean()
            ax.axhline(y=pos_ratio, color='navy', lw=2, linestyle='--',
                      label=f'Baseline = {pos_ratio:.2f}')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{tasks[task_idx]}')
            ax.legend(loc="lower left")
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{tasks[task_idx]}')
    
    # Hide unused subplots
    for i in range(n_tasks, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Precision-Recall Curves for Each Task', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curves to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train baseline model for Tox21')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['xgboost'],
                       help='Type of model to train (xgboost only for now)')
    parser.add_argument('--output_dir', type=str, default='models/baseline',
                       help='Directory to save model and results')
    parser.add_argument('--n_bits', type=int, default=2048,
                       help='Number of bits in ECFP fingerprint')
    parser.add_argument('--radius', type=int, default=2,
                       help='Radius for ECFP fingerprint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification')
    parser.add_argument('--tuned_params', type=str, default=None,
                       help='Path to tuned parameters JSON file')
    parser.add_argument('--use_spark', action='store_true',
                       help='Enable Spark featurization and distributed demo')
    args = parser.parse_args()
    
    # Load tuned parameters if provided
    tuned_params = None
    if args.tuned_params:
        try:
            with open(args.tuned_params, 'r') as f:
                tuning_results = json.load(f)
                # Extract best_params from tuning_results
                if 'best_params' in tuning_results:
                    tuned_params = tuning_results['best_params']
                elif isinstance(tuning_results, dict):
                    # Assume the file directly contains parameters
                    tuned_params = tuning_results
            print(f"✅ Loaded tuned parameters from {args.tuned_params}")
        except Exception as e:
            print(f"⚠️  Could not load tuned parameters: {e}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------- Spark session (if requested) ----------
    spark = None
    if args.use_spark and SPARK_AVAILABLE:
        spark = SparkSession.builder \
            .appName("Tox21_Baseline") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.memory", "8g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        print("🚀 Spark session created")
    elif args.use_spark and not SPARK_AVAILABLE:
        print("⚠️  PySpark not installed. Falling back to pandas.")
        args.use_spark = False
    
    try:
        # Load data
        print("Loading data...")
        train_df, val_df, test_df, train_mask, val_mask, test_mask, tasks = load_data(args.data_dir)
        
        # Prepare targets
        print("Preparing targets...")
        y_train = prepare_targets(train_df, tasks)
        y_val = prepare_targets(val_df, tasks)
        y_test = prepare_targets(test_df, tasks)
        
        # ---------- Featurization ----------
        print(f"Featurizing SMILES with ECFP (radius={args.radius}, bits={args.n_bits})...")
        start_time = time.time()
        
        X_train = featurize_ecfp(train_df['canonical_smiles'].tolist(), 
                                radius=args.radius, n_bits=args.n_bits,
                                use_spark=args.use_spark, spark_session=spark)
        X_val = featurize_ecfp(val_df['canonical_smiles'].tolist(),
                              radius=args.radius, n_bits=args.n_bits,
                              use_spark=args.use_spark, spark_session=spark)
        X_test = featurize_ecfp(test_df['canonical_smiles'].tolist(),
                               radius=args.radius, n_bits=args.n_bits,
                               use_spark=args.use_spark, spark_session=spark)
        
        end_time = time.time()
        print(f"Featurization time: {end_time - start_time:.2f}s")
        print(f"Train features shape: {X_train.shape}")
        print(f"Val features shape: {X_val.shape}")
        print(f"Test features shape: {X_test.shape}")
        
        # ---------- Spark MLlib demo (if enabled) ----------
        if args.use_spark and spark is not None:
            spark_results = train_spark_baseline(train_df, train_mask, tasks, spark, 
                                                radius=args.radius, n_bits=args.n_bits)
            # Save results for report
            spark_results_path = output_dir / 'spark_baseline_results.json'
            with open(spark_results_path, 'w') as f:
                json.dump(spark_results, f, indent=2)
            print(f"Saved Spark baseline results to {spark_results_path}")
        
        # ---------- Train XGBoost model ----------
        print(f"\nTraining {args.model_type} model...")
        
        if args.model_type == 'xgboost':
            result = train_xgboost_model(X_train, y_train, train_mask, tasks, tuned_params=tuned_params)
            model_name = 'xgboost'
        
        models = result['models']
        
        # Save models
        for task_idx, model in enumerate(models):
            if model is not None:
                model_path = output_dir / f'{model_name}_task_{task_idx}.joblib'
                dump(model, model_path)
        
        print(f"Saved models to {output_dir}")
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_result = evaluate_model(models, X_val, y_val, val_mask, tasks, args.threshold)
        val_metrics = val_result['metrics']
        
        print("\nValidation Metrics:")
        print(f"  Macro PR-AUC: {val_metrics['macro_pr_auc']:.4f}")
        print(f"  Macro ROC-AUC: {val_metrics['macro_roc_auc']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Micro PR-AUC: {val_metrics['micro_pr_auc']:.4f}")
        print(f"  Micro ROC-AUC: {val_metrics['micro_roc_auc']:.4f}")
        
        # Evaluate on test set (if not empty)
        print("\nEvaluating on test set...")
        if X_test.shape[0] > 0:
            test_result = evaluate_model(models, X_test, y_test, test_mask, tasks, args.threshold)
            test_metrics = test_result['metrics']
            y_prob_test = test_result['y_prob']
            
            # FIX: Save test probabilities for calibration
            np.save(output_dir / 'test_probs.npy', y_prob_test)
            print(f"Saved test probabilities to {output_dir / 'test_probs.npy'}")
            
            print("\nTest Metrics:")
            print(f"  Macro PR-AUC: {test_metrics['macro_pr_auc']:.4f}")
            print(f"  Macro ROC-AUC: {test_metrics['macro_roc_auc']:.4f}")
            print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            print(f"  Micro PR-AUC: {test_metrics['micro_pr_auc']:.4f}")
            print(f"  Micro ROC-AUC: {test_metrics['micro_roc_auc']:.4f}")
        else:
            print("⚠️ Test set is empty. Skipping test evaluation and saving default metrics.")
            test_metrics = {
                'macro_pr_auc': 0.0,
                'macro_roc_auc': 0.0,
                'macro_f1': 0.0,
                'micro_pr_auc': 0.0,
                'micro_roc_auc': 0.0
            }
            # Save empty test probabilities
            np.save(output_dir / 'test_probs.npy', np.empty((0, len(tasks))))
        
        # Save metrics
        metrics_dict = {
            'validation': val_metrics,
            'test': test_metrics,
            'model_type': args.model_type,
            'fingerprint': {
                'type': 'ECFP4',
                'radius': args.radius,
                'n_bits': args.n_bits
            },
            'threshold': args.threshold,
            'tuned_params_used': tuned_params is not None,
            'spark_used': args.use_spark and SPARK_AVAILABLE,
            'featurization_time_seconds': end_time - start_time
        }
        
        metrics_path = output_dir / f'{model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"\nSaved metrics to {metrics_path}")
        
        # Create visualizations (skip if no validation data)
        if X_val.shape[0] > 0:
            print("\nCreating visualizations...")
            
            # ROC curves
            roc_path = output_dir / f'{model_name}_roc_curves.png'
            plot_roc_curves(y_val, val_result['y_prob'], val_mask, tasks, str(roc_path))
            
            # PR curves
            pr_path = output_dir / f'{model_name}_pr_curves.png'
            plot_pr_curves(y_val, val_result['y_prob'], val_mask, tasks, str(pr_path))
            
            # Per-task metrics bar plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Task PR-AUC
            task_pr_aucs = [val_metrics.get(f'task_{i}_pr_auc', 0) for i in range(len(tasks))]
            axes[0].bar(range(len(tasks)), task_pr_aucs)
            axes[0].set_xlabel('Task')
            axes[0].set_ylabel('PR-AUC')
            axes[0].set_title('Per-task PR-AUC (Validation)')
            axes[0].set_xticks(range(len(tasks)))
            axes[0].set_xticklabels(tasks, rotation=45, ha='right')
            
            # Task ROC-AUC
            task_roc_aucs = [val_metrics.get(f'task_{i}_roc_auc', 0) for i in range(len(tasks))]
            axes[1].bar(range(len(tasks)), task_roc_aucs)
            axes[1].set_xlabel('Task')
            axes[1].set_ylabel('ROC-AUC')
            axes[1].set_title('Per-task ROC-AUC (Validation)')
            axes[1].set_xticks(range(len(tasks)))
            axes[1].set_xticklabels(tasks, rotation=45, ha='right')
            
            plt.tight_layout()
            task_metrics_path = output_dir / f'{model_name}_task_metrics.png'
            plt.savefig(task_metrics_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"\n✅ Baseline training completed!")
        print(f"Primary metric (macro PR-AUC):")
        print(f"  Validation: {val_metrics['macro_pr_auc']:.4f}")
        if X_test.shape[0] > 0:
            print(f"  Test: {test_metrics['macro_pr_auc']:.4f}")
        else:
            print(f"  Test: N/A (test set empty)")
        print(f"  Tuned parameters used: {tuned_params is not None}")
        print(f"  Spark mode: {args.use_spark}")
        
        # Check if baseline meets minimum performance
        if val_metrics['macro_pr_auc'] < 0.05:
            print("\n⚠️  WARNING: Baseline performance is very low (macro PR-AUC < 0.05)")
            print("  Please check data preprocessing and feature extraction.")
        
        return val_metrics['macro_pr_auc']
    
    finally:
        # Clean up Spark session
        if spark is not None:
            spark.stop()
            print("Spark session stopped")


if __name__ == '__main__':
    main()