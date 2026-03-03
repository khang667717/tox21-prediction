
#!/usr/bin/env python3
"""
Data preprocessing for Tox21 dataset.
Steps:
1. Load raw CSV (pandas or PySpark for Big Data demo)
2. Canonicalize SMILES
3. Deduplicate by canonical SMILES (keep all duplicates but log conflicts)
4. Scaffold split (80/10/10) using Bemis-Murcko scaffolds
5. Save processed CSVs and masks for each split.
6. Generate split manifest.

Optional PySpark support for Big Data demonstration.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import warnings
import os
import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# -------------------- Spark optional import --------------------
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StringType, ArrayType, DoubleType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️ PySpark not installed. Use --use_spark to enable Big Data mode.")
# -------------------------------------------------------------

# Global seed for reproducibility
SEED = 42
np.random.seed(SEED)

def canonicalize_smiles(smiles):
    """Convert SMILES to canonical SMILES."""
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_murcko_scaffold(smiles):
    """Generate Bemis-Murcko scaffold from SMILES."""
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def split_indices_by_scaffold(scaffold_list, train_ratio=0.8, val_ratio=0.1):
    """
    Split indices by scaffold so that the same scaffold goes into the same split.
    Returns three lists of indices.
    FIXED: Ensure test set has data by reserving scaffolds.
    """
    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffold_list):
        scaffold_to_indices[scaffold].append(idx)
    
    # Get list of scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    np.random.shuffle(scaffolds)
    
    total_samples = len(scaffold_list)
    
    # Calculate target sizes - use proper rounding
    target_train_samples = int(round(train_ratio * total_samples))
    target_val_samples = int(round(val_ratio * total_samples))
    target_test_samples = total_samples - target_train_samples - target_val_samples
    
    # Đảm bảo tất cả splits đều có dữ liệu
    min_samples_per_split = 100  # Tối thiểu 100 mẫu mỗi split
    
    if target_test_samples < min_samples_per_split:
        # Test set quá nhỏ, điều chỉnh lại
        target_test_samples = min(min_samples_per_split, total_samples // 5)
        target_val_samples = min(min_samples_per_split, total_samples // 5)
        target_train_samples = total_samples - target_val_samples - target_test_samples
    
    print(f"  🎯 Target sizes: Train={target_train_samples}, Val={target_val_samples}, Test={target_test_samples}")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Track sample counts
    current_train_samples = 0
    current_val_samples = 0
    
    # PHASE 1: Phân bố train set - lấy các scaffold lớn trước
    # Sắp xếp scaffolds theo kích thước giảm dần để dễ điều chỉnh
    scaffold_sizes = [(sc, len(scaffold_to_indices[sc])) for sc in scaffolds]
    scaffold_sizes.sort(key=lambda x: x[1], reverse=True)
    sorted_scaffolds = [sc for sc, _ in scaffold_sizes]
    
    remaining_scaffolds = []
    
    # Phân bố train
    for sc in sorted_scaffolds:
        indices = scaffold_to_indices[sc]
        num_samples = len(indices)
        
        if current_train_samples < target_train_samples:
            train_indices.extend(indices)
            current_train_samples += num_samples
        else:
            remaining_scaffolds.append(sc)
    
    print(f"  📊 After train: {current_train_samples}/{target_train_samples} samples, {len(remaining_scaffolds)} scaffolds remaining")
    
    # PHASE 2: Phân bố val - nhưng phải CHỪA LẠI cho test
    # Tính toán số lượng cần cho test từ scaffolds còn lại
    remaining_samples = sum(len(scaffold_to_indices[sc]) for sc in remaining_scaffolds)
    test_samples_needed = max(target_test_samples, total_samples // 10)  # Ít nhất 10%
    
    temp_remaining = []
    val_has_samples = False
    
    for sc in remaining_scaffolds:
        indices = scaffold_to_indices[sc]
        num_samples = len(indices)
        
        # Tính số lượng còn lại nếu lấy scaffold này cho val
        remaining_after_val = remaining_samples - num_samples
        
        # Nếu lấy scaffold này cho val mà không còn đủ cho test, thì bỏ qua
        if remaining_after_val < test_samples_needed * 0.8 and val_has_samples:
            temp_remaining.append(sc)
            continue
        
        # Luôn lấy scaffold đầu tiên cho val nếu val chưa có mẫu
        if not val_has_samples:
            val_indices.extend(indices)
            current_val_samples += num_samples
            val_has_samples = True
            remaining_samples -= num_samples
            print(f"  ✅ Added scaffold to val: {num_samples} samples")
            continue
        
        # Nếu đã có val và còn đủ target
        if current_val_samples < target_val_samples:
            val_indices.extend(indices)
            current_val_samples += num_samples
            remaining_samples -= num_samples
        else:
            temp_remaining.append(sc)
    
    remaining_scaffolds = temp_remaining
    
    # PHASE 3: Phần còn lại cho test
    for sc in remaining_scaffolds:
        test_indices.extend(scaffold_to_indices[sc])
    
    # Debug info
    train_scaffolds = len(set(scaffold_list[i] for i in train_indices)) if train_indices else 0
    val_scaffolds = len(set(scaffold_list[i] for i in val_indices)) if val_indices else 0
    test_scaffolds = len(set(scaffold_list[i] for i in test_indices)) if test_indices else 0
    
    print(f"\n  📈 Final split:")
    print(f"  Scaffolds: Total={len(scaffold_to_indices)}, Train={train_scaffolds}, Val={val_scaffolds}, Test={test_scaffolds}")
    print(f"  Samples: Train={len(train_indices)} ({len(train_indices)/total_samples*100:.1f}%), "
          f"Val={len(val_indices)} ({len(val_indices)/total_samples*100:.1f}%), "
          f"Test={len(test_indices)} ({len(test_indices)/total_samples*100:.1f}%)")
    
    return train_indices, val_indices, test_indices

def main():
    parser = argparse.ArgumentParser(description='Preprocess Tox21 dataset.')
    parser.add_argument('--input', type=str, default='data/raw/tox21.csv', 
                       help='Path to raw CSV')
    parser.add_argument('--output_dir', type=str, default='data/processed', 
                       help='Directory for processed files')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--use_spark', action='store_true',
                       help='Use PySpark for distributed processing (Big Data demo)')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # -----------------------------------------------------------------
    # 1. LOAD DATA – Spark or pandas
    # -----------------------------------------------------------------
    if args.use_spark and SPARK_AVAILABLE:
        print("🚀 Spark mode enabled – loading with PySpark")
        spark = SparkSession.builder \
            .appName("Tox21_Preprocess") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        df_spark = spark.read.csv(args.input, header=True, inferSchema=True)
        print(f"  Spark DataFrame rows: {df_spark.count()}")
        
        # Canonical SMILES UDF
        print("Canonicalizing SMILES with Spark...")
        canon_udf = udf(canonicalize_smiles, StringType())
        df_spark = df_spark.withColumn("canonical_smiles", canon_udf(col("smiles")))
        df_spark = df_spark.filter(col("canonical_smiles").isNotNull())
        
        # Scaffold UDF
        print("Generating Murcko scaffolds with Spark...")
        scaffold_udf = udf(get_murcko_scaffold, StringType())
        df_spark = df_spark.withColumn("scaffold", scaffold_udf(col("canonical_smiles")))
        df_spark = df_spark.filter(col("scaffold").isNotNull())
        
        # Convert to pandas for downstream processing
        df = df_spark.toPandas()
        spark.stop()
        print(f"  Converted to pandas: {len(df)} rows")
    else:
        if args.use_spark and not SPARK_AVAILABLE:
            print("⚠️ Spark requested but not available. Falling back to pandas.")
        
        print("🐼 Pandas mode – loading with pandas")
        # Load data
        print(f"Loading data from {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} rows.")
        
        # Canonicalize SMILES
        print("Canonicalizing SMILES...")
        df['canonical_smiles'] = df['smiles'].apply(canonicalize_smiles)
        
        # Drop rows with invalid SMILES
        original_len = len(df)
        df = df.dropna(subset=['canonical_smiles'])
        print(f"Dropped {original_len - len(df)} rows with invalid SMILES.")
        
        # Generate scaffolds
        print("Generating Murcko scaffolds...")
        df['scaffold'] = df['canonical_smiles'].apply(get_murcko_scaffold)
        
        # Handle NaN scaffolds (invalid molecules after scaffold generation)
        df = df.dropna(subset=['scaffold'])
        print(f"After scaffold generation: {len(df)} rows.")
    
    # Define tasks (exclude mol_id and smiles)
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
             'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
             'SR-HSE', 'SR-MMP', 'SR-p53']
    
    # Check for duplicates and conflicts
    print("Checking for duplicates and conflicts...")
    smiles_groups = df.groupby('canonical_smiles')
    conflict_rows = []
    
    for smiles, group in smiles_groups:
        if len(group) > 1:
            # Check each task for conflicts
            for task in tasks:
                non_null_vals = group[task].dropna().unique()
                if len(non_null_vals) > 1:
                    conflict_rows.append({
                        'canonical_smiles': smiles,
                        'task': task,
                        'values': list(non_null_vals),
                        'mol_ids': list(group['mol_id']) if 'mol_id' in group.columns else [],
                        'count': len(group)
                    })
    
    # Save conflict report
    if conflict_rows:
        conflict_df = pd.DataFrame(conflict_rows)
        conflict_path = Path(args.output_dir) / 'conflicts.csv'
        conflict_df.to_csv(conflict_path, index=False)
        print(f"Found {len(conflict_df)} conflicts. Saved to {conflict_path}")
    else:
        print("No conflicts found.")
    
    # Get unique scaffolds for splitting
    # We need to ensure all molecules with the same canonical_smiles go to same split
    # So we group by canonical_smiles first, take the first scaffold
    unique_smiles = df[['canonical_smiles', 'scaffold']].drop_duplicates(subset=['canonical_smiles'])
    
    # Split by scaffold
    print("Performing scaffold split (80/10/10)...")
    train_smiles_idx, val_smiles_idx, test_smiles_idx = split_indices_by_scaffold(
        unique_smiles['scaffold'].tolist(),
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # Get the canonical_smiles for each split
    train_smiles = unique_smiles.iloc[train_smiles_idx]['canonical_smiles'].tolist()
    val_smiles = unique_smiles.iloc[val_smiles_idx]['canonical_smiles'].tolist()
    test_smiles = unique_smiles.iloc[test_smiles_idx]['canonical_smiles'].tolist()
    
    # Assign all molecules with these smiles to respective splits
    train_df = df[df['canonical_smiles'].isin(train_smiles)].copy()
    val_df = df[df['canonical_smiles'].isin(val_smiles)].copy()
    test_df = df[df['canonical_smiles'].isin(test_smiles)].copy()
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save processed CSVs
    train_path = Path(args.output_dir) / 'train.csv'
    val_path = Path(args.output_dir) / 'val.csv'
    test_path = Path(args.output_dir) / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to {args.output_dir}")
    
    # Create and save masks for missing labels
    def create_mask(df, tasks):
        return df[tasks].notnull().astype(int).values
    
    train_mask = create_mask(train_df, tasks)
    val_mask = create_mask(val_df, tasks)
    test_mask = create_mask(test_df, tasks)
    
    np.save(Path(args.output_dir) / 'train_mask.npy', train_mask)
    np.save(Path(args.output_dir) / 'val_mask.npy', val_mask)
    np.save(Path(args.output_dir) / 'test_mask.npy', test_mask)
    
    # Save split manifest
    manifest = {
        'train_smiles': train_smiles,
        'val_smiles': val_smiles,
        'test_smiles': test_smiles,
        'train_indices': train_df.index.tolist(),
        'val_indices': val_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'tasks': tasks
    }
    
    manifest_path = Path(args.output_dir) / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save tasks list
    tasks_path = Path(args.output_dir) / 'tasks.json'
    with open(tasks_path, 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"✅ Preprocessing completed successfully. Spark used: {args.use_spark}")
    
    # Final validation
    train_smiles_set = set(train_smiles)
    val_smiles_set = set(val_smiles)
    test_smiles_set = set(test_smiles)
    
    # Check for overlaps
    train_val_overlap = train_smiles_set.intersection(val_smiles_set)
    train_test_overlap = train_smiles_set.intersection(test_smiles_set)
    val_test_overlap = val_smiles_set.intersection(test_smiles_set)
    
    if train_val_overlap:
        print(f"⚠️ WARNING: {len(train_val_overlap)} smiles overlap between train and val!")
    if train_test_overlap:
        print(f"⚠️ WARNING: {len(train_test_overlap)} smiles overlap between train and test!")
    if val_test_overlap:
        print(f"⚠️ WARNING: {len(val_test_overlap)} smiles overlap between val and test!")
    
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✓ No data leakage detected across splits.")

if __name__ == '__main__':
    main()