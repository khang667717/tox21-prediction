#!/usr/bin/env python3
"""
Demo sử dụng model tốt nhất đã train để dự đoán độc tính
Chạy sau khi hoàn thành pipeline với run_full_pipeline.sh
FIXED: Tương thích với GCN model, path resolution đúng, SMILES parsing tốt hơn
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
# CẤU HÌNH - THAY ĐỔI NẾU CẦN
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

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_selected_model(model_path: str):
    """Load model đã được chọn - FIXED: XGBoost multi-task"""
    print(f"\n📦 Loading model: {model_path}")
    
    model_path_obj = Path(model_path)
    
    if model_path.endswith('.pkl') or model_path_obj.suffix == '.pkl' or model_path_obj.suffix == '.joblib':
        # XGBoost model - MULTI-TASK FIX
        try:
            import joblib
            
            # Đây là đường dẫn đến một file .joblib cụ thể
            # Nhưng XGBoost có 12 files: xgboost_task_0.joblib đến xgboost_task_11.joblib
            
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
# DỰ ĐOÁN - FIXED: xử lý XGBoost đúng cách
# ============================================================================
def predict_toxicity(model, model_type, features):
    """Dự đoán độc tính cho 1 phân tử - FIXED: XGBoost multi-task"""
    
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
# LƯU KẾT QUẢ DẠNG TXT - OPTION 6
# ============================================================================
def save_option6_results_full(smiles_list, predictions_list, true_labels_list, 
                            accuracy_info, threshold, model_type, 
                            txt_file="demo_results/option6_results.txt"):
    """
    Lưu kết quả option 6 vào file TXT
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("demo_results", exist_ok=True)
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("🧪 KẾT QUẢ DỰ ĐOÁN ĐỘC TÍNH - OPTION 6\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"📌 THÔNG TIN CHUNG:\n")
        f.write(f"   - Threshold: {threshold}\n")
        f.write(f"   - Model: {model_type}\n")
        f.write(f"   - Số lượng mẫu: {len(smiles_list)}\n")
        f.write(f"   - Độ chính xác: {accuracy_info.get('accuracy', 0):.3f} ")
        f.write(f"({accuracy_info.get('correct', 0)}/{accuracy_info.get('total', 0)})\n\n")
        
        f.write("="*80 + "\n")
        f.write("📊 CHI TIẾT TỪNG PHÂN TỬ\n")
        f.write("="*80 + "\n\n")
        
        for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list, predictions_list, true_labels_list)):
            f.write(f"🔬 MOLECULE {idx+1}\n")
            f.write(f"{'─'*50}\n")
            f.write(f"SMILES: {smiles}\n\n")
            
            # Tiêu đề bảng
            f.write(f"{'Endpoint':<20} {'Prob':<8} {'Pred':<8} {'True':<8} {'Status':<8}\n")
            f.write(f"{'─'*60}\n")
            
            sorted_indices = np.argsort(preds)[::-1]
            for i in sorted_indices:
                prob = preds[i]
                pred = "TOXIC" if prob >= threshold else "SAFE"
                
                true_text = "N/A"
                status = ""
                if true_labels is not None and i < len(true_labels) and not np.isnan(true_labels[i]):
                    true_val = true_labels[i]
                    true_text = "TOXIC" if true_val == 1 else "SAFE"
                    
                    if (pred == "TOXIC" and true_text == "TOXIC") or (pred == "SAFE" and true_text == "SAFE"):
                        status = "✓"
                    else:
                        status = "✗"
                
                f.write(f"{TASK_NAMES[i]:<20} {prob:<8.4f} {pred:<8} {true_text:<8} {status:<8}\n")
            
            toxic_endpoints = np.sum(preds >= threshold)
            f.write(f"\n📌 Tóm tắt: {toxic_endpoints}/12 endpoints TOXIC\n")
            f.write(f"{'─'*60}\n\n")
        
        f.write("="*80 + "\n")
        f.write("📈 TỔNG KẾT\n")
        f.write("="*80 + "\n")
        f.write(f"- Tổng số dự đoán đúng: {accuracy_info.get('correct', 0)}/{accuracy_info.get('total', 0)}\n")
        f.write(f"- Accuracy: {accuracy_info.get('accuracy', 0):.3f}\n")
    
    print(f"\n📁 Kết quả đã được lưu vào: {txt_file}")

# ============================================================================
# LƯU KẾT QUẢ CHO CHẤT BỊ ĐÁNH GIÁ SAI - DẠNG TXT
# ============================================================================
def save_misclassified_results(smiles_list, predictions_list, true_labels_list, 
                             accuracy_info, threshold, model_type,
                             txt_file="demo_results/misclassified_results.txt"):
    """
    Lưu kết quả của các chất bị đánh giá sai vào file TXT
    NGƯỠNG SAI: 2/12
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs("demo_results", exist_ok=True)
    
    # Lọc các chất bị đánh giá sai
    misclassified = []
    error_stats = defaultdict(int)
    fp_count = 0
    fn_count = 0
    
    for idx, (smiles, preds, true_labels) in enumerate(zip(smiles_list, predictions_list, true_labels_list)):
        if true_labels is None or len(true_labels) == 0:
            continue
            
        binary_preds = (preds >= threshold).astype(int)
        
        mis_count = 0
        mis_details = []
        
        for i, task in enumerate(TASK_NAMES):
            if i < len(true_labels) and not np.isnan(true_labels[i]):
                true_val = int(true_labels[i])
                pred_val = binary_preds[i]
                
                if true_val != pred_val:
                    mis_count += 1
                    mis_details.append({
                        "endpoint": task,
                        "true": "TOXIC" if true_val == 1 else "SAFE",
                        "pred": "TOXIC" if pred_val == 1 else "SAFE",
                        "prob": float(preds[i])
                    })
                    
                    # Đếm lỗi theo endpoint
                    error_stats[task] += 1
                    
                    # Đếm FP/FN
                    if true_val == 0 and pred_val == 1:
                        fp_count += 1
                    elif true_val == 1 and pred_val == 0:
                        fn_count += 1
        
        NGUONG_SAI = 2
        if mis_count >= NGUONG_SAI:
            misclassified.append({
                "index": idx + 1,
                "smiles": smiles,
                "mis_count": mis_count,
                "details": mis_details
            })
    
    # Ghi file TXT
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("❌ CÁC CHẤT BỊ ĐÁNH GIÁ SAI\n")
        f.write("="*80 + "\n\n")
        
        f.write("📊 THÔNG TIN CHUNG\n")
        f.write(f"{'─'*40}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Ngưỡng sai: 2/12 endpoint\n")
        f.write(f"Tổng số mẫu: {len(smiles_list)}\n")
        f.write(f"Accuracy: {accuracy_info.get('accuracy', 0):.3f}\n")
        f.write(f"Số chất bị sai: {len(misclassified)}/{len(smiles_list)}\n\n")
        
        if not misclassified:
            f.write("🎉 KHÔNG CÓ CHẤT NÀO BỊ ĐÁNH GIÁ SAI!\n")
        else:
            f.write("🔬 DANH SÁCH CHI TIẾT\n")
            f.write("="*80 + "\n\n")
            
            for item in misclassified:
                f.write(f"MOLECULE {item['index']} (Sai {item['mis_count']}/12)\n")
                f.write(f"{'─'*60}\n")
                f.write(f"SMILES: {item['smiles']}\n\n")
                
                f.write(f"{'Endpoint':<20} {'True':<8} {'Pred':<8} {'Prob':<8}\n")
                f.write(f"{'─'*50}\n")
                
                for detail in item['details']:
                    f.write(f"{detail['endpoint']:<20} {detail['true']:<8} ")
                    f.write(f"{detail['pred']:<8} {detail['prob']:<8.4f}\n")
                
                f.write("\n" + "="*60 + "\n\n")
            
            # Thống kê lỗi
            f.write("\n📈 THỐNG KÊ LỖI\n")
            f.write("="*40 + "\n\n")
            
            if error_stats:
                f.write("ENDPOINT BỊ SAI NHIỀU NHẤT:\n")
                f.write(f"{'─'*30}\n")
                sorted_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)
                for endpoint, count in sorted_errors:
                    percentage = (count / len(misclassified)) * 100 if misclassified else 0
                    f.write(f"{endpoint:<20}: {count} lần ({percentage:.1f}%)\n")
                f.write("\n")
            
            total_errors = fp_count + fn_count
            if total_errors > 0:
                f.write("LOẠI LỖI:\n")
                f.write(f"{'─'*30}\n")
                f.write(f"False Positive (SAFE→TOXIC): {fp_count} ({fp_count/total_errors*100:.1f}%)\n")
                f.write(f"False Negative (TOXIC→SAFE): {fn_count} ({fn_count/total_errors*100:.1f}%)\n")
    
    print(f"\n📁 Kết quả chất bị sai đã được lưu vào: {txt_file}")

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
        print("6. Tất cả các chức năng trên")
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
            print("\n🔄 Running all demos...")
            
            # Test với các chất mẫu (hiển thị đầy đủ)
            test_with_example_compounds(model, model_type, threshold)
            
            # Test với 393 chất - CHỈ HIỂN THỊ TÓM TẮT
            print("\n📌 Đang xử lý chất random trong tập test (sẽ lưu chi tiết vào file)...")
            smiles_list, predictions_list, true_labels_list, accuracy_info = test_with_test_set(
                model, model_type, 393, threshold, return_data=True, summary_only=True
            )
            
            # Lưu kết quả đầy đủ vào file
            save_option6_results_full(
                smiles_list, predictions_list, true_labels_list, 
                accuracy_info, threshold, model_type,
                txt_file="demo_results/option6_results.txt"  # Đổi thành .txt
            )

            #Lưu kết quả các chất bị sai
            save_misclassified_results(
                smiles_list, predictions_list, true_labels_list,
                accuracy_info, threshold, model_type,
                txt_file="demo_results/misclassified_results.txt"  # Đổi thành .txt
            )
            
            # Các chức năng khác
            find_most_toxic_compounds(model, model_type, 3, threshold)
            create_toxicity_report(model, model_type, threshold)
            
            print("\n✅ All demos completed!")
            
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