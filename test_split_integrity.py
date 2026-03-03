
"""
Test for data splitting integrity - ENHANCED VERSION
Tests that no canonical_smiles appears in multiple splits.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

def test_no_data_leakage_across_splits():
    """HARD CONSTRAINT: Test that there are NO overlapping canonical_smiles between splits."""
    
    # Load split manifest
    manifest_path = Path('data/processed/split_manifest.json')
    
    if not manifest_path.exists():
        pytest.fail(f"Manifest file not found at {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    train_smiles = set(manifest['train_smiles'])
    val_smiles = set(manifest['val_smiles'])
    test_smiles = set(manifest['test_smiles'])
    
    # Check for overlaps - THIS MUST BE ZERO
    train_val_overlap = train_smiles.intersection(val_smiles)
    train_test_overlap = train_smiles.intersection(test_smiles)
    val_test_overlap = val_smiles.intersection(test_smiles)
    
    # Assert no overlaps
    assert len(train_val_overlap) == 0, \
        f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} overlapping smiles between train and val"
    
    assert len(train_test_overlap) == 0, \
        f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} overlapping smiles between train and test"
    
    assert len(val_test_overlap) == 0, \
        f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} overlapping smiles between val and test"
    
    print("✅ Test passed: No data leakage detected.")

def test_scaffold_integrity():
    """Test that all molecules with the same scaffold are in the same split."""
    
    # Load all data with scaffolds
    all_data = []
    for split_name, split_file in [('train', 'train.csv'), ('val', 'val.csv'), ('test', 'test.csv')]:
        path = Path('data/processed') / split_file
        if path.exists():
            df = pd.read_csv(path)
            df['split'] = split_name
            all_data.append(df)
    
    if not all_data:
        pytest.skip("Split files not found")
    
    all_df = pd.concat(all_data, ignore_index=True)
    
    # Group by scaffold and check splits
    scaffold_violations = []
    for scaffold, group in all_df.groupby('scaffold'):
        splits = group['split'].unique()
        # All molecules with the same scaffold should be in the same split
        if len(splits) > 1:
            scaffold_violations.append((scaffold, list(splits)))
    
    assert len(scaffold_violations) == 0, \
        f"Scaffold split violation: {len(scaffold_violations)} scaffolds appear in multiple splits"
    
    print(f"✅ Scaffold integrity test passed: {len(set(all_df['scaffold']))} unique scaffolds")

def test_mask_alignment():
    """Test that masks correctly align with NaN values in dataframes."""
    
    train_path = Path('data/processed/train.csv')
    val_path = Path('data/processed/val.csv')
    test_path = Path('data/processed/test.csv')
    
    train_mask_path = Path('data/processed/train_mask.npy')
    val_mask_path = Path('data/processed/val_mask.npy')
    test_mask_path = Path('data/processed/test_mask.npy')
    
    if not (train_path.exists() and train_mask_path.exists()):
        pytest.skip("Data or mask files not found")
    
    # Load tasks
    tasks_path = Path('data/processed/tasks.json')
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)
    
    # Test each split
    for split_name, data_path, mask_path in [
        ('train', train_path, train_mask_path),
        ('val', val_path, val_mask_path),
        ('test', test_path, test_mask_path)
    ]:
        if not data_path.exists() or not mask_path.exists():
            continue
            
        df = pd.read_csv(data_path)
        mask = np.load(mask_path)
        
        # Check shape alignment
        assert mask.shape[0] == len(df), \
            f"Mask rows {mask.shape[0]} != dataframe rows {len(df)} in {split_name}"
        
        assert mask.shape[1] == len(tasks), \
            f"Mask columns {mask.shape[1]} != tasks {len(tasks)} in {split_name}"
        
        # Verify mask matches null values
        for i, task in enumerate(tasks):
            # Mask is 1 for non-null, 0 for null
            df_null = df[task].notnull().astype(int).values
            mask_null = mask[:, i]
            
            # Check alignment
            mismatches = np.where(df_null != mask_null)[0]
            if len(mismatches) > 0:
                pytest.fail(f"Mask mismatch for task {task} in {split_name}: "
                          f"{len(mismatches)} mismatches at indices {mismatches[:5]}")
    
    print("✅ Mask alignment test passed")

def test_canonical_smiles_validity():
    """Test that all canonical_smiles can be parsed by RDKit."""
    
    from rdkit import Chem
    
    train_path = Path('data/processed/train.csv')
    val_path = Path('data/processed/val.csv')
    test_path = Path('data/processed/test.csv')
    
    invalid_smiles = []
    
    for split_name, split_path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not split_path.exists():
            continue
            
        df = pd.read_csv(split_path)
        
        for idx, smiles in enumerate(df['canonical_smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles.append((split_name, idx, smiles))
    
    assert len(invalid_smiles) == 0, \
        f"Invalid SMILES found: {len(invalid_smiles)} invalid SMILES"
    
    print("✅ All canonical SMILES are valid")

def test_split_ratios():
    """Test that split ratios are within acceptable ranges."""
    
    train_path = Path('data/processed/train.csv')
    val_path = Path('data/processed/val.csv')
    test_path = Path('data/processed/test.csv')
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Split files not found")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    # Check that splits are roughly 80/10/10
    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total
    
    # Allow some flexibility (e.g., 75-85/8-12/8-12)
    assert 0.75 <= train_ratio <= 0.85, \
        f"Train ratio {train_ratio:.3f} outside expected range [0.75, 0.85]"
    
    assert 0.08 <= val_ratio <= 0.12, \
        f"Val ratio {val_ratio:.3f} outside expected range [0.08, 0.12]"
    
    assert 0.08 <= test_ratio <= 0.12, \
        f"Test ratio {test_ratio:.3f} outside expected range [0.08, 0.12]"
    
    print(f"✅ Split ratios: Train={train_ratio:.3f}, Val={val_ratio:.3f}, Test={test_ratio:.3f}")

def test_manifest_consistency():
    """Test that manifest is consistent with saved CSV files."""
    
    manifest_path = Path('data/processed/split_manifest.json')
    
    if not manifest_path.exists():
        pytest.skip("Manifest file not found")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Check each split
    for split_name, data_path in [
        ('train', 'train.csv'),
        ('val', 'val.csv'),
        ('test', 'test.csv')
    ]:
        path = Path('data/processed') / data_path
        if not path.exists():
            continue
            
        df = pd.read_csv(path)
        manifest_smiles = set(manifest[f'{split_name}_smiles'])
        csv_smiles = set(df['canonical_smiles'])
        
        # Check exact match
        assert manifest_smiles == csv_smiles, \
            f"Manifest mismatch for {split_name}: " \
            f"{len(manifest_smiles - csv_smiles)} SMILES in manifest but not CSV, " \
            f"{len(csv_smiles - manifest_smiles)} SMILES in CSV but not manifest"
    
    print("✅ Manifest consistency test passed")

if __name__ == '__main__':
    # Run tests directly
    import sys
    
    tests = [
        test_no_data_leakage_across_splits,
        test_scaffold_integrity,
        test_mask_alignment,
        test_canonical_smiles_validity,
        test_split_ratios,
        test_manifest_consistency
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            print(f"  ✅ PASSED")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed_tests.append(test.__name__)
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed: {failed_tests}")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(tests)} tests passed!")