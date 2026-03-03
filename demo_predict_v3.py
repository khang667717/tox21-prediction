#!/usr/bin/env python3
"""
Demo sử dụng model tốt nhất đã train để dự đoán độc tính
Chạy sau khi hoàn thành pipeline với run_full_pipeline.sh
ENHANCED: Kết hợp tinh hoá của cả 2 phiên bản
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# Import các module từ project
from scripts.feature_extraction import get_canonical_smiles
from scripts.feature_extraction import (
    extract_morgan_fingerprint,
    extract_morgan_fingerprint_counts,
    smiles_to_graph
)

# ============================================================================
# CẤU HÌNH - DỄ DÀNG THAY ĐỔI
# ============================================================================
def find_model_file(model_name: str) -> str:
    """
    Tìm đường dẫn thực tế của model file.
    Args:
        model_name: Tên model từ best_model_selection.json
    Returns:
        Path đến model file thực tế
    """
    # Mapping từ tên model trong JSON đến đường dẫn thực tế
    model_paths = {
        'gnn_gcn_model.pt': 'models/gnn_gcn/final_model.pt',
        'gnn_gat_model.pt': 'models/gnn_gat/final_model.pt',
        'descriptor_mlp_model.pt': 'models/descriptor_mlp/final_model.pt',
        'xgboost_model.pkl': 'models/baseline/xgboost_task_0.joblib',
    }
    
    # Nếu có trong mapping, dùng đường dẫn đã biết
    if model_name in model_paths:
        path = Path(model_paths[model_name])
        if path.exists():
            print(f"✅ Found model at: {path}")
            return str(path)
    
    # Thử symlink trước
    if 'gcn' in model_name.lower() or 'gnn' in model_name.lower():
        symlink_path = Path('models/gnn_model.pt')
        if symlink_path.exists():
            try:
                real_path = symlink_path.resolve()
                if real_path.exists():
                    print(f"✅ Found model via symlink: {real_path}")
                    return str(real_path)
            except:
                pass
    
    # Tìm trong các thư mục con
    for subdir in ['gnn_gcn', 'gnn_gat', 'descriptor_mlp', 'baseline']:
        candidate = Path(f'models/{subdir}/final_model.pt')
        if candidate.exists() and (subdir in model_name.lower() or 'gnn' in model_name.lower()):
            return str(candidate)
    
    # Thử tìm file .joblib cho baseline
    if 'xgboost' in model_name.lower() or 'baseline' in model_name.lower():
        pkl_files = list(Path('models/baseline').glob('*.joblib'))
        if pkl_files:
            return str(pkl_files[0])
    
    # Fallback: đường dẫn gốc
    return str(Path('models') / model_name)

# Load cấu hình từ pipeline
try:
    with open('best_model_selection.json', 'r') as f:
        model_info = json.load(f)
    SELECTED_MODEL = model_info.get('selected_model', 'gnn_gcn_model.pt')
    MODEL_PATH = find_model_file(SELECTED_MODEL)
except FileNotFoundError:
    print("⚠️ best_model_selection.json not found. Looking for available models...")
    # Tìm model khả dụng
    found_models = []
    for subdir in ['gnn_gcn', 'gnn_gat', 'descriptor_mlp']:
        model_file = Path(f'models/{subdir}/final_model.pt')
        if model_file.exists():
            found_models.append((str(model_file), subdir))
    
    if found_models:
        MODEL_PATH, SELECTED_MODEL = found_models[0][0], f"{found_models[0][1]}_model.pt"
        print(f"✅ Using fallback model: {MODEL_PATH}")
    else:
        # Tìm file XGBoost bất kỳ
        xgb_files = list(Path('models/baseline').glob('xgboost_task_*.joblib'))
        if xgb_files:
            xgb_files.sort()
            MODEL_PATH = str(xgb_files[-1])  # Lấy file có số lớn nhất
            SELECTED_MODEL = 'xgboost_model.pkl'
            print(f"✅ Found XGBoost model: {MODEL_PATH}")
        else:
            MODEL_PATH = 'models/baseline/xgboost_task_7.joblib'
            SELECTED_MODEL = 'xgboost_model.pkl'
            print(f"⚠️ Using default XGBoost path: {MODEL_PATH}")

TEST_DATA_PATH = 'data/processed/test.csv'

TASK_NAMES = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Trọng số endpoints dựa trên tài liệu Tox21
ENDPOINT_WEIGHTS = {
    'NR-AR': 1.2,        # Androgen receptor - quan trọng
    'NR-AR-LBD': 1.0,
    'NR-AhR': 1.3,       # Aryl hydrocarbon receptor - rất quan trọng
    'NR-Aromatase': 1.1,
    'NR-ER': 1.5,        # Estrogen receptor - CỰC KỲ QUAN TRỌNG
    'NR-ER-LBD': 1.2,
    'NR-PPAR-gamma': 1.0,
    'SR-ARE': 1.3,       # Stress response - quan trọng
    'SR-ATAD5': 1.2,     # Genotoxicity - rất quan trọng
    'SR-HSE': 1.0,
    'SR-MMP': 1.4,       # Mitochondrial toxicity - rất quan trọng
    'SR-p53': 1.5,       # DNA damage - CỰC KỲ QUAN TRỌNG
}

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_selected_model(model_path: str):
    """Load model đã được chọn - Hỗ trợ XGBoost multi-task và PyTorch models"""
    print(f"\n📦 Loading model: {model_path}")
    
    model_path_obj = Path(model_path)
    
    if model_path.endswith('.pkl') or model_path_obj.suffix == '.pkl' or model_path_obj.suffix == '.joblib':
        # XGBoost model - MULTI-TASK
        try:
            import joblib
            
            # Lấy thư mục chứa model
            model_dir = Path(model_path).parent
            
            # Tìm tất cả các file task
            task_files = sorted(model_dir.glob('xgboost_task_*.joblib'))
            
            if len(task_files) == 12:  # Có đủ 12 tasks
                print(f"✅ Found {len(task_files)} XGBoost task files")
                models = []
                for task_file in task_files:
                    try:
                        models.append(joblib.load(task_file))
                    except Exception as e:
                        print(f"⚠️ Failed to load {task_file}: {e}")
                        models.append(None)
                model = models  # List of 12 models
                model_type = 'xgboost'
                print(f"✅ XGBoost multi-task model loaded: {len(models)} tasks")
            else:
                # Fallback: load single file
                print(f"⚠️ Only found {len(task_files)} task files, loading single model")
                model = joblib.load(model_path)
                model_type = 'xgboost'
                
        except Exception as e:
            print(f"❌ Failed to load XGBoost model: {e}")
            raise
        
    elif model_path.endswith('.pt') or model_path_obj.suffix == '.pt':
        # PyTorch model (MLP/GCN/GAT)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise
        
        # Kiểm tra xem model từ gnn_model.py hay từ models/
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"   Model config: {config.keys()}")
            
            # Nếu là GNN model từ gnn_model.py
            if 'gnn_type' in config:
                # Import từ gnn_model.py
                import sys
                if 'scripts' not in sys.path:
                    sys.path.append('scripts')
                from scripts.gnn_model import GNNModel
                
                model = GNNModel(
                    node_feature_dim=config.get('node_feature_dim', 43),
                    edge_feature_dim=config.get('edge_feature_dim', 4),
                    hidden_dim=config.get('hidden_dim', 128),
                    num_layers=config.get('num_layers', 3),
                    gnn_type=config.get('gnn_type', 'gcn'),
                    num_heads=config.get('num_heads', 4),
                    dropout_rate=config.get('dropout_rate', 0.2),
                    output_dim=config.get('output_dim', 12),
                    use_edge_features=config.get('use_edge_features', False),
                    pooling=config.get('pooling', 'mean')
                )
                model_type = config.get('gnn_type', 'gcn')  # 'gcn' hoặc 'gat'
                print(f"   GNN model type: {model_type}")
                
            else:
                # MLP từ descriptor_mlp.py
                import sys
                if 'scripts' not in sys.path:
                    sys.path.append('scripts')
                from scripts.descriptor_mlp import MLPModel
                
                model = MLPModel(
                    input_dim=config.get('input_dim', 184),
                    hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                    output_dim=config.get('output_dim', 12),
                    dropout_rate=config.get('dropout_rate', 0.2)
                )
                model_type = 'mlp'
                print(f"   MLP model loaded")
        else:
            # Fallback - dựa vào tên file
            print(f"   No model_config found, guessing model type from filename...")
            if 'mlp' in model_path.lower():
                from scripts.descriptor_mlp import MLPModel
                model = MLPModel(
                    input_dim=184,
                    hidden_dims=[256, 128, 64],
                    output_dim=12,
                    dropout_rate=0.2
                )
                model_type = 'mlp'
            elif 'gcn' in model_path.lower():
                import sys
                if 'scripts' not in sys.path:
                    sys.path.append('scripts')
                from scripts.gnn_model import GNNModel
                model = GNNModel(
                    node_feature_dim=43,
                    edge_feature_dim=4,
                    hidden_dim=128,
                    num_layers=3,
                    gnn_type='gcn',
                    num_heads=4,
                    dropout_rate=0.2,
                    output_dim=12,
                    use_edge_features=False,
                    pooling='mean'
                )
                model_type = 'gcn'
            elif 'gat' in model_path.lower():
                import sys
                if 'scripts' not in sys.path:
                    sys.path.append('scripts')
                from scripts.gnn_model import GNNModel
                model = GNNModel(
                    node_feature_dim=43,
                    edge_feature_dim=4,
                    hidden_dim=128,
                    num_layers=3,
                    gnn_type='gat',
                    num_heads=4,
                    dropout_rate=0.2,
                    output_dim=12,
                    use_edge_features=False,
                    pooling='mean'
                )
                model_type = 'gat'
            else:
                raise ValueError(f"Cannot determine model type from {model_path}")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model weights loaded successfully")
        model.eval()
        
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    
    print(f"✅ Model ready for inference")
    print(f"   Type: {model_type}")
    print(f"   Path: {model_path}")
    
    return model, model_type

# ============================================================================
# CHUẨN BỊ FEATURES
# ============================================================================
def prepare_features(smiles: str, model_type: str):
    """Chuẩn bị features tương thích với model"""
    
    # 1. Get canonical SMILES với error handling tốt hơn
    try:
        canonical_smiles = get_canonical_smiles(smiles)
        if not canonical_smiles:
            # Thử parse với sanitize=False
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                except:
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        raise ValueError(f"Invalid SMILES: {smiles} - {e}")
    
    if not canonical_smiles:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 2. Prepare features theo model type
    if model_type == 'xgboost':
        # Morgan fingerprint cho XGBoost
        features = extract_morgan_fingerprint_counts(canonical_smiles)
        features = np.array(features).reshape(1, -1)
        
    elif model_type in ['mlp']:
        # MLP: Morgan fingerprint
        features = extract_morgan_fingerprint(canonical_smiles)
        features = torch.FloatTensor(features).unsqueeze(0)
        
    elif model_type in ['gcn', 'gat']:
        # GNN: Graph features
        from torch_geometric.data import Data
        
        graph_dict = smiles_to_graph(canonical_smiles)
        if graph_dict is None:
            raise ValueError(f"Cannot convert SMILES to graph: {canonical_smiles}")
        
        # Tạo PyTorch Geometric Data object
        features = Data(
            x=torch.FloatTensor(graph_dict['x']),
            edge_index=torch.LongTensor(graph_dict['edge_index']),
            edge_attr=torch.FloatTensor(graph_dict['edge_attr']),
            batch=torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return features, canonical_smiles

# ============================================================================
# DỰ ĐOÁN - XỬ LÝ XGBoost ĐÚNG CÁCH
# ============================================================================
def predict_toxicity(model, model_type, features):
    """Dự đoán độc tính cho 1 phân tử"""
    
    try:
        if model_type == 'xgboost':
            # XGBoost multi-task prediction
            if isinstance(model, list):
                # Model là list các XGBoost classifiers (12 tasks)
                predictions = np.zeros(len(TASK_NAMES), dtype=np.float32)
                
                for task_idx, task_model in enumerate(model):
                    if task_idx >= len(TASK_NAMES):
                        break
                        
                    if task_model is not None:
                        try:
                            if hasattr(task_model, 'predict_proba'):
                                proba = task_model.predict_proba(features)
                                if len(proba.shape) == 2 and proba.shape[1] >= 2:
                                    predictions[task_idx] = proba[0, 1]  # Xác suất class positive
                                else:
                                    predictions[task_idx] = proba[0, 0]
                            else:
                                predictions[task_idx] = task_model.predict(features)[0]
                        except Exception as e:
                            print(f"⚠️ Error predicting task {TASK_NAMES[task_idx]}: {e}")
                            predictions[task_idx] = 0.0
                    else:
                        predictions[task_idx] = 0.0
                
                if not hasattr(predict_toxicity, '_printed_xgb'):
                    print(f"   📊 XGBoost multi-task: {sum(m is not None for m in model)}/{len(model)} tasks active")
                    predict_toxicity._printed_xgb = True
            else:
                # Fallback: model đơn
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    if len(proba.shape) == 2 and proba.shape[1] >= 2:
                        pred_value = proba[0, 1]
                    else:
                        pred_value = proba[0, 0]
                    predictions = np.full(len(TASK_NAMES), pred_value, dtype=np.float32)
                else:
                    pred_value = model.predict(features)[0]
                    predictions = np.full(len(TASK_NAMES), float(pred_value), dtype=np.float32)
        
        elif model_type in ['mlp']:
            with torch.no_grad():
                output = model(features)
                predictions = torch.sigmoid(output).numpy()[0]
                
        elif model_type in ['gcn', 'gat']:
            with torch.no_grad():
                output = model(features)
                predictions = torch.sigmoid(output).numpy()[0]
                
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in predict_toxicity: {e}")
        import traceback
        traceback.print_exc()
        raise
    
# ============================================================================
# PHÂN TÍCH KẾT QUẢ
# ============================================================================
def analyze_predictions(smiles_list: List[str], predictions: np.ndarray, 
                       true_labels: np.ndarray = None, threshold: float = 0.5):
    """Phân tích và hiển thị kết quả dự đoán"""
    
    print("\n" + "="*80)
    print("🧪 TOXICITY PREDICTION RESULTS")
    print("="*80)
    
    for idx, (smiles, preds) in enumerate(zip(smiles_list, predictions)):
        print(f"\n📊 Molecule {idx+1}: {smiles[:60]}...")
        print("-" * 60)
        
        # Sắp xếp các endpoint theo độ tin cậy (probability cao nhất)
        sorted_indices = np.argsort(preds)[::-1]
        
        print(f"{'Toxicity Endpoint':<20} {'Probability':<12} {'Prediction':<12} {'True Label':<12}")
        print("-" * 60)
        
        for i in sorted_indices[:6]:  # Hiển thị top 6
            prob = preds[i]
            pred = "TOXIC" if prob >= threshold else "SAFE"
            
            true_label = ""
            if true_labels is not None:
                true_val = true_labels[idx, i]
                if not np.isnan(true_val):
                    true_label = "TOXIC" if true_val == 1 else "SAFE"
                    if (pred == "TOXIC" and true_label == "TOXIC") or (pred == "SAFE" and true_label == "SAFE"):
                        true_label = f"✓ {true_label}"
                    else:
                        true_label = f"✗ {true_label}"
            
            print(f"{TASK_NAMES[i]:<20} {prob:<12.4f} {pred:<12} {true_label:<12}")
        
        # Summary
        toxic_endpoints = np.sum(preds >= threshold)
        print(f"\n🔬 Summary: {toxic_endpoints}/{len(TASK_NAMES)} endpoints predicted as toxic")
        
        # Tìm endpoint độc nhất
        max_tox = np.max(preds)
        if max_tox >= 0.7:
            toxic_idx = np.argmax(preds)
            print(f"⚠️  Highest toxicity: {TASK_NAMES[toxic_idx]} ({max_tox:.3f})")

# ============================================================================
# TEST VỚI CÁC CHẤT MẪU
# ============================================================================
def test_with_example_compounds(model, model_type, threshold=0.5):
    """Test với các chất mẫu an toàn và độc hại"""
    
    # Các chất mẫu từ Tox21 test set
    example_compounds = {
        "Very Toxic (Aflatoxin B1)": "O=C1C=CC2=C(O1)C3=C(C=C2)C4=C(C(=O)OC4)C5=C3OCO5",
        "Toxic (Aspirin)": "CC(=O)Oc1ccccc1C(=O)O",
        "Moderately Toxic (Bisphenol A)": "CC(C)(c1ccc(cc1)O)c2ccc(cc2)O",
        "Relatively Safe (Ethanol)": "CCO",
        "Very Safe (Sucrose)": "C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O",
        "Known Safe (Vitamin C)": "OC1C(OC(C(C1O)O)O)C(=O)O",
        "Benzene (Carcinogen)": "c1ccccc1",
    }
    
    print("\n" + "="*80)
    print("🧪 TESTING WITH EXAMPLE COMPOUNDS")
    print("="*80)
    
    all_predictions = []
    all_smiles = []
    
    for compound_name, smiles in example_compounds.items():
        print(f"\n🔬 Testing: {compound_name}")
        print(f"   SMILES: {smiles}")
        
        try:
            # Chuẩn bị features
            features, canonical_smiles = prepare_features(smiles, model_type)
            
            # Dự đoán
            predictions = predict_toxicity(model, model_type, features)
            
            # Lưu kết quả
            all_predictions.append(predictions)
            all_smiles.append(canonical_smiles)
            
            # Hiển thị kết quả nhanh
            toxic_count = np.sum(predictions >= threshold)
            max_tox = np.max(predictions)
            print(f"   📊 Results: {toxic_count} toxic endpoints, max toxicity: {max_tox:.3f}")
            
            if max_tox >= 0.5:
                toxic_idx = np.argmax(predictions)
                print(f"   ⚠️  Most toxic: {TASK_NAMES[toxic_idx]} ({max_tox:.3f})")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_predictions.append(np.full(len(TASK_NAMES), np.nan))
            all_smiles.append(smiles)
    
    # Phân tích tổng hợp
    analyze_predictions(all_smiles, np.array(all_predictions), threshold=threshold)

# ============================================================================
# TEST VỚI DỮ LIỆU TEST SET
# ============================================================================
def test_with_test_set(model, model_type, n_samples: int = 10, threshold=0.5, 
                       return_data=False, summary_only=False):
    """Test với ngẫu nhiên n_samples từ test set"""
    
    print("\n" + "="*80)
    print(f"📊 TESTING WITH {n_samples} RANDOM COMPOUNDS FROM TEST SET")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Chọn ngẫu nhiên n_samples
    if n_samples > len(test_df):
        n_samples = len(test_df)
    
    sample_indices = np.random.choice(len(test_df), n_samples, replace=False)
    sample_df = test_df.iloc[sample_indices].copy()
    
    all_predictions = []
    all_smiles = []
    all_true_labels = []
    
    if summary_only:
        print(f"\n🔍 Đang xử lý {n_samples} chất (chỉ hiển thị tóm tắt)...")
    else:
        print(f"\n🔍 Selected {n_samples} random compounds:")
        print("-" * 60)
    
    for idx, row in sample_df.iterrows():
        smiles = row['canonical_smiles']
        
        if not summary_only:
            print(f"{idx+1:2d}. {smiles[:60]}...")
        
        try:
            # Chuẩn bị features
            features, canonical_smiles = prepare_features(smiles, model_type)
            
            # Dự đoán
            predictions = predict_toxicity(model, model_type, features)
            
            # Lưu kết quả
            all_predictions.append(predictions)
            all_smiles.append(canonical_smiles)
            all_true_labels.append(row[TASK_NAMES].values)
            
        except Exception as e:
            if not summary_only:
                print(f"   ❌ Error: {e}")
            all_predictions.append(np.full(len(TASK_NAMES), np.nan))
            all_smiles.append(smiles)
            all_true_labels.append(np.full(len(TASK_NAMES), np.nan))
    
    # Chỉ hiển thị phân tích chi tiết nếu không phải summary_only
    if not summary_only:
        analyze_predictions(all_smiles, np.array(all_predictions), 
                          np.array(all_true_labels), threshold)
    
    # Tính accuracy
    accuracy_info = {}
    all_true_labels_array = np.array(all_true_labels)
    valid_mask = ~np.isnan(all_true_labels_array.astype(float))
    
    if np.any(valid_mask):
        predictions_array = np.array(all_predictions)
        true_array = np.array(all_true_labels)
        
        binary_preds = (predictions_array >= threshold).astype(float)
        
        correct = (binary_preds[valid_mask] == true_array[valid_mask]).sum()
        total = valid_mask.sum()
        accuracy = correct / total if total > 0 else 0
        
        accuracy_info = {
            "accuracy": float(accuracy),
            "correct": int(correct),
            "total": int(total)
        }
        
        print(f"\n📈 Sample Accuracy: {accuracy:.3f} ({correct}/{total} correct predictions)")
    
    if return_data:
        return all_smiles, all_predictions, all_true_labels, accuracy_info
    
# ============================================================================
# TÌM KIẾM CHẤT NGUY HIỂM NHẤT
# ============================================================================
def find_most_toxic_compounds(model, model_type, n: int = 5, threshold=0.5):
    """Tìm n chất nguy hiểm nhất trong test set"""
    
    print("\n" + "="*80)
    print(f"⚠️  FINDING {n} MOST TOXIC COMPOUNDS IN TEST SET")
    print("="*80)
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    print(f"\n🔍 Screening {len(test_df)} compounds...")
    
    toxicity_scores = []
    compound_info = []
    
    # Duyệt qua một phần test set để tiết kiệm thời gian
    max_screen = min(200, len(test_df))
    sample_indices = np.random.choice(len(test_df), max_screen, replace=False)
    
    for i, idx in enumerate(sample_indices):
        if i % 50 == 0:
            print(f"   Processed {i}/{max_screen} compounds...")
        
        smiles = test_df.iloc[idx]['canonical_smiles']
        
        try:
            # Chuẩn bị features
            features, canonical_smiles = prepare_features(smiles, model_type)
            
            # Dự đoán
            predictions = predict_toxicity(model, model_type, features)
            
            # Tính toxicity score: trung bình probability + penalty cho high toxic endpoints
            avg_toxicity = np.mean(predictions)
            high_tox_count = np.sum(predictions >= 0.8)  # Rất độc
            toxicity_score = avg_toxicity + (high_tox_count * 0.1)
            
            toxicity_scores.append(toxicity_score)
            compound_info.append({
                'smiles': canonical_smiles,
                'predictions': predictions,
                'avg_toxicity': avg_toxicity,
                'high_tox_count': high_tox_count
            })
            
        except Exception as e:
            toxicity_scores.append(0)
            compound_info.append(None)
    
    # Tìm n chất độc nhất
    valid_indices = [i for i, score in enumerate(toxicity_scores) if score > 0]
    if not valid_indices:
        print("❌ No valid predictions found!")
        return
    
    top_indices = np.argsort(toxicity_scores)[-n:][::-1]
    top_indices = [i for i in top_indices if i in valid_indices][:n]
    
    print(f"\n🏆 TOP {len(top_indices)} MOST TOXIC COMPOUNDS:")
    print("-" * 80)
    
    for rank, idx in enumerate(top_indices, 1):
        if compound_info[idx] is not None:
            info = compound_info[idx]
            smiles = info['smiles']
            avg_tox = info['avg_toxicity']
            high_count = info['high_tox_count']
            
            print(f"\n#{rank} - Average Toxicity: {avg_tox:.3f}")
            print(f"   SMILES: {smiles}")
            print(f"   High toxicity endpoints (>0.8): {high_count}")
            
            # Tìm endpoint độc nhất
            predictions = info['predictions']
            top_endpoint_idx = np.argmax(predictions)
            top_tox = predictions[top_endpoint_idx]
            print(f"   Most toxic: {TASK_NAMES[top_endpoint_idx]} ({top_tox:.3f})")
            
            # Hiển thị top 3 toxic endpoints
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            for j, tox_idx in enumerate(top_3_idx, 1):
                print(f"   {j}. {TASK_NAMES[tox_idx]}: {predictions[tox_idx]:.3f}")

# ============================================================================
# TẠO BÁO CÁO VISUALIZATION
# ============================================================================
def create_toxicity_report(model, model_type, threshold=0.5):
    """Tạo báo cáo visualization"""
    
    print("\n" + "="*80)
    print("📊 CREATING TOXICITY PREDICTION REPORT")
    print("="*80)
    
    # Load một số chất từ test set
    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_df = test_df.sample(n=20, random_state=42).reset_index(drop=True)
    
    # Tính predictions
    predictions_list = []
    compound_names = []
    valid_indices = []
    
    for idx, row in sample_df.iterrows():
        smiles = row['canonical_smiles']
        
        try:
            features, canonical_smiles = prepare_features(smiles, model_type)
            preds = predict_toxicity(model, model_type, features)
            predictions_list.append(preds)
            compound_names.append(canonical_smiles[:30] + "...")
            valid_indices.append(idx)
        except Exception as e:
            print(f"   ⚠️ Error processing {smiles}: {e}")
            predictions_list.append(np.full(len(TASK_NAMES), np.nan))
            compound_names.append("ERROR")
    
    predictions_array = np.array(predictions_list)
    
    # Tạo heatmap
    plt.figure(figsize=(14, 10))
    
    # Tạo mask cho NaN values
    mask = np.isnan(predictions_array)
    
    # Plot heatmap
    plt.imshow(np.where(mask, 0, predictions_array), 
               cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Thêm colorbar
    plt.colorbar(label='Toxicity Probability')
    
    # Labels
    plt.yticks(range(len(compound_names)), compound_names, fontsize=9)
    plt.xticks(range(len(TASK_NAMES)), TASK_NAMES, rotation=45, 
               ha='right', fontsize=9)
    
    plt.title(f'Toxicity Predictions - Model: {model_type} (threshold={threshold})', fontsize=14)
    plt.tight_layout()
    
    # Lưu figure
    os.makedirs('demo_results', exist_ok=True)
    plt.savefig('demo_results/toxicity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig('demo_results/toxicity_heatmap.pdf', bbox_inches='tight')
    
    print(f"\n✅ Report saved to:")
    print(f"   - demo_results/toxicity_heatmap.png")
    print(f"   - demo_results/toxicity_heatmap.pdf")
    
    # Tạo summary statistics
    avg_toxicity = np.nanmean(predictions_array, axis=0)
    
    print(f"\n📈 AVERAGE TOXICITY BY ENDPOINT:")
    print("-" * 40)
    for i, (task, avg) in enumerate(zip(TASK_NAMES, avg_toxicity)):
        print(f"{task:<20}: {avg:.3f}")
    
    # Lưu predictions với SMILES
    if len(predictions_list) > 0:
        report_df = pd.DataFrame(predictions_array, columns=TASK_NAMES)
        report_df['SMILES'] = sample_df.iloc[valid_indices]['canonical_smiles'].values
        report_df.to_csv('demo_results/toxicity_predictions.csv', index=False)
        print(f"\n📄 Detailed predictions saved to: demo_results/toxicity_predictions.csv")

# ============================================================================
# DỰ ĐOÁN SMILES NHẬP TAY
# ============================================================================
def predict_single_smiles(model, model_type, smiles: str, threshold: float = 0.5):
    """Dự đoán cho một SMILES duy nhất"""
    
    try:
        features, canonical_smiles = prepare_features(smiles, model_type)
        predictions = predict_toxicity(model, model_type, features)
        
        print(f"\n🔬 Predictions for: {canonical_smiles}")
        print("-" * 50)
        
        # Sắp xếp theo probability giảm dần
        sorted_indices = np.argsort(predictions)[::-1]
        
        for i in sorted_indices:
            task = TASK_NAMES[i]
            prob = predictions[i]
            pred = "TOXIC" if prob >= threshold else "SAFE"
            print(f"{task:<20}: {prob:.4f} ({pred})")
        
        # Summary
        toxic_count = np.sum(predictions >= threshold)
        print(f"\n📊 Summary: {toxic_count}/{len(TASK_NAMES)} endpoints predicted as toxic")
        
        # Most toxic endpoint
        max_tox = np.max(predictions)
        if max_tox >= threshold:
            toxic_idx = np.argmax(predictions)
            print(f"⚠️  Most toxic: {TASK_NAMES[toxic_idx]} ({max_tox:.3f})")
        
        return predictions, canonical_smiles
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# LƯU KẾT QUẢ DẠNG README VÀ JSON (ENHANCED VERSION)
# ============================================================================
def save_option6_results_full(smiles_list, predictions_list, true_labels_list, 
                            accuracy_info, threshold, model_type, 
                            json_file="demo_results/option6_results.json",
                            readme_file="demo_results/option6_results.md"):
    """
    Lưu kết quả option 6 vào cả JSON và README.md
    ENHANCED: Xử lý lỗi NaN an toàn và phân tích đa chiều
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("demo_results", exist_ok=True)
    
    # === TÍNH TOÁN THÔNG TIN TỔNG HỢP ===
    total_molecules = len(smiles_list)
    correct_molecules = 0
    incorrect_molecules = 0
    molecule_results = []
    
    for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list, predictions_list, true_labels_list)):
        # Xử lý true_labels an toàn
        if true_labels is not None and len(true_labels) > 0:
            try:
                # Chuyển sang float array, thay NaN = -1
                true_labels_float = np.array(true_labels, dtype=np.float32)
                
                # Tạo mask cho valid values (không phải NaN và không phải -1)
                valid_mask = ~np.isnan(true_labels_float)
                
                if np.any(valid_mask):
                    binary_preds = (preds >= threshold).astype(float)
                    
                    # So sánh chỉ trên các giá trị valid
                    correct_endpoints = (binary_preds[valid_mask] == true_labels_float[valid_mask]).sum()
                    total_endpoints = valid_mask.sum()
                    
                    # Tính weighted score dựa trên tầm quan trọng của endpoint
                    total_weight = 0
                    correct_weight = 0
                    for i, task in enumerate(TASK_NAMES):
                        if valid_mask[i]:
                            weight = ENDPOINT_WEIGHTS.get(task, 1.0)
                            total_weight += weight
                            if binary_preds[i] == true_labels_float[i]:
                                correct_weight += weight
                    
                    weighted_score = correct_weight / total_weight if total_weight > 0 else 0
                    
                    # Nếu tất cả endpoints đều đúng hoặc chỉ sai 1 thì molecule đúng
                    if correct_endpoints >= total_endpoints - 1:  # Cho phép sai 1
                        molecule_result = "ĐÚNG"
                        correct_molecules += 1
                    else:
                        molecule_result = "SAI"
                        incorrect_molecules += 1

                    molecule_results.append({
                        "index": idx + 1,
                        "smiles": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                        "correct_endpoints": int(correct_endpoints),
                        "total_endpoints": int(total_endpoints),
                        "accuracy": correct_endpoints / total_endpoints if total_endpoints > 0 else 0,
                        "weighted_score": float(weighted_score),
                        "result": molecule_result
                    })
                else:
                    # Không có valid endpoints
                    molecule_results.append({
                        "index": idx + 1,
                        "smiles": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                        "correct_endpoints": 0,
                        "total_endpoints": 0,
                        "accuracy": 0,
                        "weighted_score": 0,
                        "result": "KHÔNG CÓ LABEL"
                    })
            except Exception as e:
                print(f"⚠️ Lỗi xử lý molecule {idx+1}: {e}")
                molecule_results.append({
                    "index": idx + 1,
                    "smiles": smiles[:50] + "..." if len(smiles) > 50 else smiles,
                    "correct_endpoints": 0,
                    "total_endpoints": 0,
                    "accuracy": 0,
                    "weighted_score": 0,
                    "result": "LỖI XỬ LÝ"
                })
    
    # === LƯU FILE JSON (dữ liệu thô) ===
    results_json = {
        "threshold": threshold,
        "model_type": model_type,
        "total_molecules": total_molecules,
        "accuracy": accuracy_info,
        "summary": {
            "total_molecules": total_molecules,
            "correct_molecules": correct_molecules,
            "incorrect_molecules": incorrect_molecules,
            "accuracy_molecules": correct_molecules / total_molecules if total_molecules > 0 else 0,
            "correct_predictions": accuracy_info.get('correct', 0),
            "total_predictions": accuracy_info.get('total', 0),
            "accuracy_endpoints": accuracy_info.get('accuracy', 0)
        },
        "molecule_summary": molecule_results,
        "predictions": []
    }
    
    for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list, predictions_list, true_labels_list)):
        molecule_result = {
            "index": idx + 1,
            "smiles": smiles,
            "predictions": {},
            "true_labels": {},
            "summary": {
                "toxic_endpoints": int(np.sum(preds >= threshold)),
                "max_toxicity": float(np.max(preds)),
                "max_toxic_endpoint": TASK_NAMES[np.argmax(preds)] if np.max(preds) >= threshold else None
            }
        }
        
        for i, task in enumerate(TASK_NAMES):
            molecule_result["predictions"][task] = {
                "probability": float(preds[i]),
                "prediction": "TOXIC" if preds[i] >= threshold else "SAFE"
            }
            
            # Xử lý true_labels an toàn
            if true_labels is not None and i < len(true_labels):
                val = true_labels[i]
                # Chỉ xử lý nếu là số
                if isinstance(val, (int, float)) and not np.isnan(float(val)):
                    molecule_result["true_labels"][task] = {
                        "value": int(val),
                        "label": "TOXIC" if val == 1 else "SAFE"
                    }
        
        results_json["predictions"].append(molecule_result)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # === LƯU FILE README.md (dễ đọc) ===
    with open(readme_file, 'w', encoding='utf-8') as f:
        # === THÔNG TIN TỔNG QUAN ===
        f.write("# 🧪 KẾT QUẢ DỰ ĐOÁN ĐỘC TÍNH - OPTION 6\n\n")
        f.write("## 📊 THÔNG TIN TỔNG QUAN\n\n")
        f.write(f"- **Threshold**: {threshold}\n")
        f.write(f"- **Model**: {model_type}\n")
        f.write(f"- **Tổng số mẫu**: {total_molecules}\n\n")
        
        # === THỐNG KÊ CHI TIẾT ===
        f.write("## 📈 THỐNG KÊ CHI TIẾT\n\n")
        f.write("### 📊 THEO ENDPOINT (tổng số dự đoán)\n")
        f.write(f"- **Tổng số dự đoán**: {accuracy_info.get('total', 0)}\n")
        f.write(f"- **Số dự đoán đúng**: {accuracy_info.get('correct', 0)}\n")
        f.write(f"- **Số dự đoán sai**: {accuracy_info.get('total', 0) - accuracy_info.get('correct', 0)}\n")
        f.write(f"- **Độ chính xác (endpoint)**: {accuracy_info.get('accuracy', 0):.3f}\n\n")
        
        f.write("### 🧪 THEO PHÂN TỬ (molecule-level accuracy)\n")
        f.write(f"- **Tổng số phân tử**: {total_molecules}\n")
        f.write(f"- **Số phân tử đúng (sai ≤1 endpoint)**: {correct_molecules}\n")
        f.write(f"- **Số phân tử sai (sai ≥2 endpoints)**: {incorrect_molecules}\n")
        f.write(f"- **Độ chính xác (phân tử)**: {correct_molecules/total_molecules:.3f}\n\n")
        
        # === DANH SÁCH PHÂN TỬ ĐÚNG/SAI ===
        if correct_molecules > 0:
            f.write("## ✅ DANH SÁCH PHÂN TỬ ĐÚNG\n\n")
            correct_list = [m for m in molecule_results if m["result"] == "ĐÚNG"]
            for m in correct_list[:10]:  # Chỉ hiển thị 10 đầu
                f.write(f"- **Molecule {m['index']}**: `{m['smiles']}`\n")
            if len(correct_list) > 10:
                f.write(f"- ... và {len(correct_list) - 10} phân tử khác\n")
            f.write(f"\n*Tổng số: {len(correct_list)} phân tử*\n\n")
        
        if incorrect_molecules > 0:
            f.write("## ❌ DANH SÁCH PHÂN TỬ SAI\n\n")
            incorrect_list = [m for m in molecule_results if m["result"] == "SAI"]
            for m in incorrect_list[:10]:
                f.write(f"- **Molecule {m['index']}**: `{m['smiles']}` (đúng {m['correct_endpoints']}/{m['total_endpoints']} endpoints, {m['accuracy']*100:.1f}%)\n")
            if len(incorrect_list) > 10:
                f.write(f"- ... và {len(incorrect_list) - 10} phân tử khác\n")
            f.write(f"\n*Tổng số: {len(incorrect_list)} phân tử*\n\n")
        
        # === CHI TIẾT 5 PHÂN TỬ ĐẦU TIÊN ===
        f.write("## 🔬 CHI TIẾT 5 PHÂN TỬ ĐẦU TIÊN\n\n")
        
        for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list[:5], predictions_list[:5], true_labels_list[:5])):
            f.write(f"### Molecule {idx+1}\n")
            f.write(f"**SMILES**: `{smiles}`\n\n")
            
            # Tính số endpoint đúng cho molecule này
            if true_labels is not None and len(true_labels) > 0:
                try:
                    true_labels_float = np.array(true_labels, dtype=np.float32)
                    valid_mask = ~np.isnan(true_labels_float)
                    
                    if np.any(valid_mask):
                        binary_preds = (preds >= threshold).astype(float)
                        correct_endpoints = (binary_preds[valid_mask] == true_labels_float[valid_mask]).sum()
                        total_endpoints = valid_mask.sum()
                        
                        # Tính weighted score
                        total_weight = 0
                        correct_weight = 0
                        for i, task in enumerate(TASK_NAMES):
                            if valid_mask[i]:
                                weight = ENDPOINT_WEIGHTS.get(task, 1.0)
                                total_weight += weight
                                if binary_preds[i] == true_labels_float[i]:
                                    correct_weight += weight
                        
                        weighted_score = correct_weight / total_weight if total_weight > 0 else 0
                        
                        f.write(f"**Kết quả**: {correct_endpoints}/{total_endpoints} endpoints đúng ")
                        if correct_endpoints == total_endpoints:
                            f.write("✅ HOÀN HẢO\n")
                        elif correct_endpoints >= total_endpoints - 1:
                            f.write("✓ TỐT\n")
                        else:
                            f.write(f"❌ CẦN CẢI THIỆN\n")
                        f.write(f"**Weighted Score**: {weighted_score:.3f}\n\n")
                except:
                    f.write("**Kết quả**: Không thể tính toán\n\n")
            
            # Tạo bảng markdown
            f.write("| Toxicity Endpoint | Probability | Prediction | True Label |\n")
            f.write("|-------------------|-------------|------------|------------|\n")
            
            sorted_indices = np.argsort(preds)[::-1]
            for i in sorted_indices:
                prob = preds[i]
                pred = "TOXIC" if prob >= threshold else "SAFE"
                
                true_label_text = ""
                if true_labels is not None and i < len(true_labels):
                    val = true_labels[i]
                    if isinstance(val, (int, float)) and not np.isnan(float(val)):
                        true_label_text = "TOXIC" if val == 1 else "SAFE"
                        if (pred == "TOXIC" and true_label_text == "TOXIC") or (pred == "SAFE" and true_label_text == "SAFE"):
                            true_label_text = f"✓ {true_label_text}"
                        else:
                            true_label_text = f"✗ {true_label_text}"
                
                f.write(f"| {TASK_NAMES[i]:<19} | {prob:.4f} | {pred:<10} | {true_label_text:<10} |\n")
            
            toxic_endpoints = np.sum(preds >= threshold)
            f.write(f"\n**Tóm tắt**: {toxic_endpoints}/12 endpoints dự đoán là TOXIC\n\n")
            f.write("---\n\n")
        
        # === PHÂN TÍCH ĐA CHIỀU ===
        f.write("## 📊 PHÂN TÍCH ĐA CHIỀU\n\n")
        
        # 1. PHÂN LOẠI THEO CHẤT LƯỢNG
        f.write("### 📈 1. PHÂN LOẠI THEO CHẤT LƯỢNG\n\n")
        
        excellent = sum(1 for m in molecule_results if m["total_endpoints"] > 0 and m["accuracy"] >= 0.9)
        good = sum(1 for m in molecule_results if m["total_endpoints"] > 0 and 0.8 <= m["accuracy"] < 0.9)
        fair = sum(1 for m in molecule_results if m["total_endpoints"] > 0 and 0.7 <= m["accuracy"] < 0.8)
        poor = sum(1 for m in molecule_results if m["total_endpoints"] > 0 and m["accuracy"] < 0.7)
        
        f.write(f"- **Xuất sắc (≥90%)**: {excellent} phân tử ({excellent/total_molecules*100:.1f}%)\n")
        f.write(f"- **Tốt (80-89%)**: {good} phân tử ({good/total_molecules*100:.1f}%)\n")
        f.write(f"- **Khá (70-79%)**: {fair} phân tử ({fair/total_molecules*100:.1f}%)\n")
        f.write(f"- **Kém (<70%)**: {poor} phân tử ({poor/total_molecules*100:.1f}%)\n\n")
        
        # 2. PHÂN LOẠI THEO SỐ ENDPOINT SAI
        f.write("### 🎯 2. PHÂN LOẠI THEO SỐ ENDPOINT SAI\n\n")
        
        perfect = sum(1 for m in molecule_results if m["correct_endpoints"] == m["total_endpoints"])
        one_error = sum(1 for m in molecule_results if m["total_endpoints"] - m["correct_endpoints"] == 1)
        two_errors = sum(1 for m in molecule_results if m["total_endpoints"] - m["correct_endpoints"] == 2)
        three_plus = sum(1 for m in molecule_results if m["total_endpoints"] - m["correct_endpoints"] >= 3)
        
        f.write(f"- **Hoàn hảo (0 sai)**: {perfect} phân tử ({perfect/total_molecules*100:.1f}%)\n")
        f.write(f"- **Sai 1 endpoint**: {one_error} phân tử ({one_error/total_molecules*100:.1f}%)\n")
        f.write(f"- **Sai 2 endpoints**: {two_errors} phân tử ({two_errors/total_molecules*100:.1f}%)\n")
        f.write(f"- **Sai ≥3 endpoints**: {three_plus} phân tử ({three_plus/total_molecules*100:.1f}%)\n\n")
        
        # 3. WEIGHTED SCORE
        f.write("### ⚖️ 3. WEIGHTED SCORE (THEO ĐỘ QUAN TRỌNG)\n\n")
        
        f.write("**Trọng số endpoints:**\n")
        for task, weight in ENDPOINT_WEIGHTS.items():
            if weight >= 1.4:
                f.write(f"  - **{task}**: {weight} (RẤT QUAN TRỌNG)\n")
        f.write("\n")
        
        weighted_scores = [m["weighted_score"] for m in molecule_results if m["total_endpoints"] > 0]
        if weighted_scores:
            avg_weighted = sum(weighted_scores) / len(weighted_scores)
            f.write(f"- **Weighted Score trung bình**: {avg_weighted:.3f}\n")
            f.write(f"- **Ý nghĩa**: Tính đến độ quan trọng của từng endpoint\n\n")
        
        # 4. TỔNG HỢP ĐÁNH GIÁ
        f.write("### 🏆 4. TỔNG HỢP ĐÁNH GIÁ\n\n")
        
        # Tính điểm tổng hợp (1-10)
        endpoint_acc = accuracy_info.get('accuracy', 0) * 10  # 0-10
        perfect_pct = perfect / total_molecules * 10
        weighted_avg = avg_weighted * 10 if weighted_scores else endpoint_acc
        
        composite_score = (endpoint_acc * 0.5 + perfect_pct * 0.3 + weighted_avg * 0.2)
        
        f.write(f"- **Điểm endpoint accuracy**: {endpoint_acc:.1f}/10\n")
        f.write(f"- **Điểm hoàn hảo**: {perfect_pct:.1f}/10\n")
        f.write(f"- **Điểm weighted**: {weighted_avg:.1f}/10\n")
        f.write(f"- **ĐIỂM TỔNG HỢP**: {composite_score:.1f}/10 - ")
        
        if composite_score >= 8.5:
            f.write("🏆 **XUẤT SẮC**\n")
        elif composite_score >= 7.0:
            f.write("🥇 **RẤT TỐT**\n")
        elif composite_score >= 5.5:
            f.write("🥈 **KHÁ TỐT**\n")
        elif composite_score >= 4.0:
            f.write("🥉 **TRUNG BÌNH**\n")
        else:
            f.write("📉 **CẦN CẢI THIỆN**\n")
        
        # 5. THỐNG KÊ THEO ENDPOINT (phân tích lỗi)
        f.write("\n### 🔍 5. PHÂN TÍCH LỖI THEO ENDPOINT\n\n")
        
        # Đếm số lần mỗi endpoint bị sai
        error_counts = defaultdict(int)
        fp_counts = defaultdict(int)  # False Positive
        fn_counts = defaultdict(int)  # False Negative
        
        for idx, (preds, true_labels) in enumerate(zip(predictions_list, true_labels_list)):
            if true_labels is not None and len(true_labels) > 0:
                true_labels_float = np.array(true_labels, dtype=np.float32)
                valid_mask = ~np.isnan(true_labels_float)
                binary_preds = (preds >= threshold).astype(float)
                
                for i, task in enumerate(TASK_NAMES):
                    if valid_mask[i]:
                        if binary_preds[i] != true_labels_float[i]:
                            error_counts[task] += 1
                            if binary_preds[i] == 1 and true_labels_float[i] == 0:
                                fp_counts[task] += 1
                            elif binary_preds[i] == 0 and true_labels_float[i] == 1:
                                fn_counts[task] += 1
        
        if error_counts:
            f.write("| Endpoint | Tổng lỗi | False Positive | False Negative |\n")
            f.write("|----------|----------|----------------|----------------|\n")
            
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            for endpoint, total in sorted_errors:
                fp = fp_counts.get(endpoint, 0)
                fn = fn_counts.get(endpoint, 0)
                f.write(f"| {endpoint} | {total} | {fp} | {fn} |\n")
            
            f.write("\n**Kết luận:** ")
            if error_counts:
                worst_endpoint = max(error_counts.items(), key=lambda x: x[1])
                f.write(f"Endpoint khó dự đoán nhất là **{worst_endpoint[0]}** với {worst_endpoint[1]} lỗi.\n")
        
        # === TỔNG KẾT CUỐI CÙNG ===
        f.write("\n## 📊 TỔNG KẾT CUỐI CÙNG\n\n")
        f.write(f"- **Tổng số phân tử**: {total_molecules}\n")
        f.write(f"- **Phân tử đúng**: {correct_molecules}\n")
        f.write(f"- **Phân tử sai**: {incorrect_molecules}\n")
        f.write(f"- **Độ chính xác (phân tử)**: {correct_molecules/total_molecules:.3f}\n")
        f.write(f"- **Độ chính xác (endpoint)**: {accuracy_info.get('accuracy', 0):.3f}\n")
        f.write(f"- **Điểm tổng hợp**: {composite_score:.1f}/10\n")
    
    print(f"\n📁 Kết quả đã được lưu vào:")
    print(f"   - {json_file}")
    print(f"   - {readme_file}")
    print(f"\n📊 TỔNG KẾT NHANH:")
    print(f"   - Tổng số phân tử: {total_molecules}")
    print(f"   - Phân tử đúng: {correct_molecules}")
    print(f"   - Phân tử sai: {incorrect_molecules}")
    print(f"   - Độ chính xác (phân tử): {correct_molecules/total_molecules:.3f}")
    print(f"   - Độ chính xác (endpoint): {accuracy_info.get('accuracy', 0):.3f}")

# ============================================================================
# LƯU KẾT QUẢ CHO CHẤT BỊ ĐÁNH GIÁ SAI
# ============================================================================
def save_misclassified_results(smiles_list, predictions_list, true_labels_list, 
                             accuracy_info, threshold, model_type,
                             json_file="demo_results/misclassified_results.json",
                             readme_file="demo_results/misclassified_results.md"):
    """
    Lưu kết quả của các chất bị đánh giá sai vào JSON và README.md
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("demo_results", exist_ok=True)
    
    # Lọc các chất bị đánh giá sai
    misclassified = []
    misclassified_indices = []
    
    for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list, predictions_list, true_labels_list)):
        if true_labels is None or len(true_labels) == 0:
            continue
            
        # Chuyển predictions thành binary
        binary_preds = (preds >= threshold).astype(int)
        
        # Kiểm tra từng endpoint
        is_misclassified = False
        mis_details = []
        
        for i, task in enumerate(TASK_NAMES):
            if i < len(true_labels) and not np.isnan(true_labels[i]):
                true_val = int(true_labels[i])
                pred_val = binary_preds[i]
                
                if true_val != pred_val:
                    is_misclassified = True
                    mis_details.append({
                        "endpoint": task,
                        "true": "TOXIC" if true_val == 1 else "SAFE",
                        "predicted": "TOXIC" if pred_val == 1 else "SAFE",
                        "probability": float(preds[i])
                    })
        
        if is_misclassified:
            misclassified_indices.append(idx)
            misclassified.append({
                "index": idx + 1,
                "smiles": smiles,
                "details": mis_details,
                "summary": {
                    "total_endpoints": len(TASK_NAMES),
                    "misclassified_count": len(mis_details),
                    "accuracy": f"{(len(TASK_NAMES) - len(mis_details))}/{len(TASK_NAMES)}"
                }
            })
    
    if not misclassified:
        print("\n✅ Không có chất nào bị đánh giá sai!")
        # Tạo file thông báo
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write("# 🎉 KHÔNG CÓ CHẤT NÀO BỊ ĐÁNH GIÁ SAI\n\n")
            f.write(f"- **Threshold**: {threshold}\n")
            f.write(f"- **Model**: {model_type}\n")
            f.write(f"- **Tổng số mẫu**: {len(smiles_list)}\n")
            f.write(f"- **Accuracy**: {accuracy_info.get('accuracy', 0):.3f}\n\n")
            f.write("✨ Tất cả các dự đoán đều chính xác! ✨\n")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "threshold": threshold,
                "model_type": model_type,
                "total_samples": len(smiles_list),
                "accuracy": accuracy_info,
                "misclassified_count": 0,
                "message": "Không có chất nào bị đánh giá sai"
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Đã tạo file thông báo:")
        print(f"   - {json_file}")
        print(f"   - {readme_file}")
        return
    
    # === LƯU FILE JSON ===
    results_json = {
        "threshold": threshold,
        "model_type": model_type,
        "total_samples": len(smiles_list),
        "accuracy": accuracy_info,
        "misclassified_count": len(misclassified),
        "misclassified": misclassified
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # === LƯU FILE README.md ===
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("# ❌ CÁC CHẤT BỊ ĐÁNH GIÁ SAI\n\n")
        f.write(f"## 📊 THÔNG TIN CHUNG\n\n")
        f.write(f"- **Threshold**: {threshold}\n")
        f.write(f"- **Model**: {model_type}\n")
        f.write(f"- **Tổng số mẫu**: {len(smiles_list)}\n")
        f.write(f"- **Accuracy**: {accuracy_info.get('accuracy', 0):.3f}\n")
        f.write(f"- **Số chất bị sai**: {len(misclassified)}/{len(smiles_list)}\n\n")
        
        if misclassified:
            f.write("## 🔬 DANH SÁCH CHI TIẾT\n\n")
            
            for item in misclassified:
                f.write(f"### 🔴 Molecule {item['index']}\n")
                f.write(f"**SMILES**: `{item['smiles']}`\n\n")
                
                # Tạo bảng markdown
                f.write("| Endpoint | True Label | Predicted | Probability | Status |\n")
                f.write("|----------|------------|-----------|-------------|--------|\n")
                
                # Sắp xếp theo probability giảm dần để dễ nhìn
                details_sorted = sorted(item['details'], key=lambda x: x['probability'], reverse=True)
                
                for detail in details_sorted:
                    status = "❌ FALSE"
                    f.write(f"| {detail['endpoint']:<19} | {detail['true']:<10} | {detail['predicted']:<9} | {detail['probability']:.4f} | {status:<5} |\n")
                
                f.write(f"\n**Tóm tắt**: {item['summary']['accuracy']} endpoints đúng\n\n")
                f.write("---\n\n")
        
        # Thống kê theo loại lỗi
        f.write("## 📈 THỐNG KÊ LỖI\n\n")
        
        # Đếm số lần mỗi endpoint bị sai
        error_counts = defaultdict(int)
        for item in misclassified:
            for detail in item['details']:
                error_counts[detail['endpoint']] += 1
        
        if error_counts:
            f.write("### Endpoint bị sai nhiều nhất:\n\n")
            f.write("| Endpoint | Số lần sai |\n")
            f.write("|----------|------------|\n")
            
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            for endpoint, count in sorted_errors:
                f.write(f"| {endpoint} | {count} |\n")
            
            f.write("\n")
        
        # Thống kê loại lỗi
        fp_count = 0  # False Positive
        fn_count = 0  # False Negative
        
        for item in misclassified:
            for detail in item['details']:
                if detail['true'] == "SAFE" and detail['predicted'] == "TOXIC":
                    fp_count += 1
                elif detail['true'] == "TOXIC" and detail['predicted'] == "SAFE":
                    fn_count += 1
        
        f.write("### Loại lỗi:\n\n")
        f.write(f"- **False Positive** (SAFE → TOXIC): {fp_count}\n")
        f.write(f"- **False Negative** (TOXIC → SAFE): {fn_count}\n")
    
    print(f"\n📁 Kết quả chất bị sai đã được lưu vào:")
    print(f"   - {json_file}")
    print(f"   - {readme_file}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main function với menu lựa chọn"""
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        TOX21 MODEL DEMO - PREDICTION TOOL               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"📊 Selected Model: {SELECTED_MODEL}")
    print(f"📁 Model Path: {MODEL_PATH}")
    print("")

    # Load model một lần duy nhất
    try:
        model, model_type = load_selected_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Set threshold
    while True:
        try:
            thresh_input = input("Nhập threshold cho predictions (0-1, mặc định: 0.5): ") or "0.5"
            threshold = float(thresh_input)
            if 0 <= threshold <= 1:
                break
            else:
                print("❌ Threshold phải từ 0 đến 1!")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ (ví dụ: 0.45)")
        
    while True:
        print("\n" + "="*60)
        print("🎯 SELECT DEMO MODE:")
        print("="*60)
        print("1. Test với các chất mẫu (an toàn/độc hại)")
        print("2. Test với ngẫu nhiên từ test set")
        print("3. Tìm chất nguy hiểm nhất trong test set")
        print("4. Tạo báo cáo visualization")
        print("5. Dự đoán độc tính cho SMILES nhập tay")
        print("6. Tất cả các chức năng trên (ENHANCED - có phân tích đa chiều)")
        print("0. Thoát")
        print("-" * 60)
        
        choice = input("👉 Lựa chọn của bạn (0-6): ").strip()
        
        if choice == '1':
            test_with_example_compounds(model, model_type, threshold)
            
        elif choice == '2':
            try:
                n = int(input("Nhập số lượng mẫu (mặc định: 10): ") or "10")
                test_with_test_set(model, model_type, n, threshold)
            except ValueError:
                print("❌ Số lượng không hợp lệ!")
                
        elif choice == '3':
            try:
                n = int(input("Nhập số chất nguy hiểm nhất cần tìm (mặc định: 5): ") or "5")
                find_most_toxic_compounds(model, model_type, n, threshold)
            except ValueError:
                print("❌ Số lượng không hợp lệ!")
                
        elif choice == '4':
            create_toxicity_report(model, model_type, threshold)
            
        elif choice == '5':
            smiles = input("Nhập SMILES cần dự đoán: ").strip()
            if smiles:
                predict_single_smiles(model, model_type, smiles, threshold)
            else:
                print("❌ Vui lòng nhập SMILES!")
                
        elif choice == '6':
            print("\n🔄 Running all demos with ENHANCED analysis...")
            
            # Test với các chất mẫu (hiển thị đầy đủ)
            test_with_example_compounds(model, model_type, threshold)
            
            # Test với 300 chất - CHỈ HIỂN THỊ TÓM TẮT
            print("\n📌 Đang xử lý chất random trong tập test (sẽ lưu chi tiết vào file)...")
            smiles_list, predictions_list, true_labels_list, accuracy_info = test_with_test_set(
                model, model_type, 300, threshold, return_data=True, summary_only=True
            )
            
            # Lưu kết quả đầy đủ vào file (đã tích hợp phân tích đa chiều)
            save_option6_results_full(
                smiles_list, 
                predictions_list, 
                true_labels_list, 
                accuracy_info,
                threshold, 
                model_type,
                json_file="demo_results/option6_results.json",
                readme_file="demo_results/option6_results.md"
            )

            # Lưu kết quả các chất bị sai
            save_misclassified_results(
                smiles_list,
                predictions_list,
                true_labels_list,
                accuracy_info,
                threshold,
                model_type,
                json_file="demo_results/misclassified_results.json",
                readme_file="demo_results/misclassified_results.md"
            )
            
            # Các chức năng khác
            find_most_toxic_compounds(model, model_type, 3, threshold)
            create_toxicity_report(model, model_type, threshold)
            
            print("\n✅ All demos completed with enhanced analysis!")
            
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Lựa chọn không hợp lệ!")
        
        input("\n📌 Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    # Kiểm tra model đã tồn tại chưa
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}!")
        print(f"   Please run the pipeline first: ./run_full_pipeline.sh")
        
        # Tìm model khả dụng
        found_models = []
        
        # Tìm trong các thư mục con
        for subdir in ['gnn_gcn', 'gnn_gat', 'descriptor_mlp', 'baseline']:
            model_file = Path(f'models/{subdir}/final_model.pt')
            if model_file.exists():
                found_models.append((str(model_file), subdir))
        
        # Tìm file .joblib trong models/baseline
        baseline_models = list(Path('models/baseline').glob('*.joblib'))
        for m in baseline_models:
            found_models.append((str(m), 'baseline'))
        
        if found_models:
            print(f"\n📁 Found existing models:")
            for i, (m, subdir) in enumerate(found_models[:5], 1):
                print(f"   {i}. {m}")
            
            use_existing = input("\nUse the first available model? (y/n): ").lower()
            if use_existing == 'y':
                MODEL_PATH = found_models[0][0]
                SELECTED_MODEL = Path(MODEL_PATH).name
                print(f"✅ Using: {SELECTED_MODEL}")
                print(f"   Path: {MODEL_PATH}")
                main()
        else:
            print("\n💡 Please train models first by running:")
            print("   ./run_full_pipeline.sh")
    else:
        main()