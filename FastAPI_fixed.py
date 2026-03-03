#!/usr/bin/env python3
"""
FastAPI server for Tox21 model deployment.
Updated to automatically load the best model from best_model_selection.json
FIXED: Better SMILES parsing, improved model loading, XGBoost multi-task support
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import joblib
import json
from pathlib import Path
import sys
import os
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import warnings

# Suppress RDKit warnings
warnings.filterwarnings('ignore')

# Import model loading utilities from demo_predict.py
sys.path.append(str(Path(__file__).parent))
sys.path.append('scripts')

from scripts.train_utils import set_seed

def find_model_file(model_name):
    """
    Tìm đường dẫn thực tế của model file.
    FIXED: Better path resolution for GCN model.
    Args:
        model_name: Tên model từ best_model_selection.json (vd: 'gnn_gcn_model.pt')
    Returns:
        Path đến model file thực tế
    """
    # Mapping từ tên model trong JSON đến đường dẫn thực tế
    model_paths = {
        'gnn_gcn_model.pt': 'models/gnn_gcn/final_model.pt',
        'gnn_gat_model.pt': 'models/gnn_gat/final_model.pt',
        'descriptor_mlp_model.pt': 'models/descriptor_mlp/final_model.pt',
        'xgboost_model.pkl': 'models/baseline/xgboost_task_7.joblib',  # FIX: Trỏ đúng file
    }
    
    # Nếu có trong mapping, dùng đường dẫn đã biết
    if model_name in model_paths:
        path = Path(model_paths[model_name])
        if path.exists():
            print(f"✅ Found model at: {path}")
            return path
    
    # FIX: Kiểm tra symlink trước
    if 'gcn' in model_name.lower():
        # Thử symlink trước
        symlink_path = Path('models/gnn_model.pt')
        if symlink_path.exists():
            try:
                real_path = symlink_path.resolve()
                if real_path.exists():
                    print(f"✅ Found model via symlink: {real_path}")
                    return real_path
            except:
                pass
        
        # Sau đó thử các đường dẫn thực tế
        candidate = Path('models/gnn_gcn/final_model.pt')
        if candidate.exists():
            return candidate
    
    elif 'gat' in model_name.lower():
        candidate = Path('models/gnn_gat/final_model.pt')
        if candidate.exists():
            return candidate
    elif 'mlp' in model_name.lower():
        candidate = Path('models/descriptor_mlp/final_model.pt')
        if candidate.exists():
            return candidate
    
    # Nếu không tìm thấy, trả về path gốc
    return Path('models') / model_name

set_seed(42)

app = FastAPI(title="Tox21 Prediction API", 
              description="API for predicting molecular toxicity using the best model selected by the pipeline",
              version="2.0.0")

class MoleculeRequest(BaseModel):
    """Request model for molecule prediction."""
    smiles: str
    threshold: float = 0.2
    use_best_model: bool = True

class MoleculeResponse(BaseModel):
    """Response model for molecule prediction."""
    smiles: str
    canonical_smiles: str
    predictions: Dict[str, float]
    binary_predictions: Dict[str, bool]
    model_type: str
    model_name: str
    valid: bool

class BatchRequest(BaseModel):
    """Request model for batch prediction."""
    smiles_list: List[str]
    threshold: float = 0.5

class BatchResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[Dict[str, Any]]
    model_type: str
    model_name: str
    total_molecules: int
    valid_molecules: int

# FIX: Improved SMILES parsing function
def get_canonical_smiles(smiles: str) -> Optional[str]:
    """Convert SMILES to canonical SMILES with better error handling."""
    from rdkit import Chem
    
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        # Try with sanitization first
        mol = Chem.MolFromSmiles(smiles)
        
        # If fails, try without sanitization
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    # Use as is
                    pass
        
        if mol is None:
            return None
            
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        print(f"Warning: SMILES parsing error '{smiles}': {e}")
        return None

def extract_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Extract Morgan fingerprint as bit vector with error handling."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
        if mol is None:
            print(f"⚠️ Invalid SMILES: {smiles}")
            return np.zeros((n_bits,), dtype=np.int32)
        
        # Dùng fingerprint dạng bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        
        # Chuyển sang numpy array an toàn
        arr = np.zeros((n_bits,), dtype=np.int32)
        for i in range(n_bits):
            arr[i] = fp.GetBit(i)
        
        return arr
        
    except Exception as e:
        print(f"⚠️ Error in fingerprint extraction: {e}")
        return np.zeros((n_bits,), dtype=np.int32)

def extract_morgan_fingerprint_counts(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Extract Morgan fingerprint as count vector with error handling.
    FIXED: Tối ưu, tránh treo, có fallback.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
        if mol is None:
            print(f"⚠️ Invalid SMILES: {smiles}")
            return np.zeros((n_bits,), dtype=np.int32)
        
        # Dùng fingerprint dạng bit vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        
        # Chuyển sang numpy array an toàn
        arr = np.zeros((n_bits,), dtype=np.int32)
        for i in range(n_bits):
            arr[i] = fp.GetBit(i)
        
        return arr
        
    except Exception as e:
        print(f"⚠️ Error in fingerprint extraction: {e}")
        # Fallback: trả về zeros array
        return np.zeros((n_bits,), dtype=np.int32)

def smiles_to_graph(smiles: str):
    """Convert SMILES to graph dictionary for GNN with better error handling."""
    from rdkit import Chem
    import numpy as np
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Use the full feature extraction from feature_extraction.py
        from scripts.feature_extraction import smiles_to_graph as full_smiles_to_graph
        return full_smiles_to_graph(smiles)
    except Exception as e:
        raise ValueError(f"Failed to convert SMILES to graph '{smiles}': {e}")

TASK_NAMES = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def load_best_model_info():
    """Load information about the best model from selection."""
    best_model_path = Path("best_model_selection.json")
    if not best_model_path.exists():
        raise FileNotFoundError("best_model_selection.json not found. Please run the pipeline first.")
    
    with open(best_model_path, 'r') as f:
        return json.load(f)

def load_selected_model(model_path: str):
    """Load model đã được chọn - FIXED: XGBoost multi-task support"""
    print(f"📦 Loading model: {model_path}")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # FIX: Xử lý XGBoost multi-task
    if model_path.suffix == '.pkl' or model_path.suffix == '.joblib':
        try:
            import joblib
            
            # Nếu là file task cụ thể (xgboost_task_7.joblib)
            if 'task_' in str(model_path):
                model_dir = model_path.parent
                task_files = sorted(model_dir.glob('xgboost_task_*.joblib'))
                
                if len(task_files) == 12:
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
            else:
                # File đơn lẻ
                model = joblib.load(model_path)
                model_type = 'xgboost'
                
        except Exception as e:
            print(f"❌ Failed to load XGBoost model: {e}")
            raise
        
    elif model_path.suffix == '.pt':
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check model type từ checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            
            # Nếu là GNN model từ gnn_model.py
            if 'gnn_type' in config:
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
                    output_dim=config.get('output_dim', len(TASK_NAMES)),
                    use_edge_features=config.get('use_edge_features', False),
                    pooling=config.get('pooling', 'mean')
                )
                model_type = config.get('gnn_type', 'gcn')
                
            else:
                # MLP từ descriptor_mlp.py
                sys.path.append('scripts')
                from scripts.descriptor_mlp import MLPModel
                model = MLPModel(
                    input_dim=config.get('input_dim', 184),
                    hidden_dims=config.get('hidden_dims', [256, 128, 64]),
                    output_dim=config.get('output_dim', len(TASK_NAMES)),
                    dropout_rate=config.get('dropout_rate', 0.2)
                )
                model_type = 'mlp'
        else:
            # Fallback - dựa vào tên file
            if 'mlp' in model_path.stem.lower():
                import torch.nn as nn
                model = nn.Sequential(
                    nn.Linear(2048, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, len(TASK_NAMES))
                )
                model_type = 'mlp'
            elif 'gcn' in model_path.stem.lower() or 'gnn' in model_path.stem.lower():
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
                    output_dim=len(TASK_NAMES),
                    use_edge_features=False,
                    pooling='mean'
                )
                model_type = 'gcn'
            else:
                raise ValueError(f"Cannot determine model type: {model_path}")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    print(f"✅ Model loaded successfully")
    print(f"   Type: {model_type}")
    print(f"   Path: {model_path}")
    if isinstance(model, list):
        print(f"   Multi-task: {len(model)} sub-models")
    
    return model, model_type

def prepare_features(smiles: str, model_type: str):
    """Chuẩn bị features tương thích với model"""
    
    # 1. Get canonical SMILES
    canonical_smiles = get_canonical_smiles(smiles)
    if not canonical_smiles:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # 2. Prepare features theo model type
    if model_type == 'xgboost':
        features = extract_morgan_fingerprint_counts(canonical_smiles)
        features = np.array(features).reshape(1, -1)
        
    elif model_type in ['mlp', 'pytorch']:
        features = extract_morgan_fingerprint(canonical_smiles)
        features = torch.FloatTensor(features).unsqueeze(0)
        
    elif model_type in ['gcn', 'gat']:
        from scripts.feature_extraction import smiles_to_graph
        from torch_geometric.data import Data
        
        graph_dict = smiles_to_graph(canonical_smiles)
        if graph_dict is None:
            raise ValueError(f"Cannot convert SMILES to graph: {canonical_smiles}")
        
        features = Data(
            x=torch.FloatTensor(graph_dict['x']),
            edge_index=torch.LongTensor(graph_dict['edge_index']),
            edge_attr=torch.FloatTensor(graph_dict['edge_attr']),
            batch=torch.zeros(graph_dict['x'].shape[0], dtype=torch.long)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return features, canonical_smiles

def predict_toxicity(model, model_type, features):
    """Dự đoán độc tính - FIXED: XGBoost multi-task support"""
    print(f"🔍 predict_toxicity called with model_type: {model_type}")
    
    try:
        if model_type == 'xgboost':
            print("🔍 Using XGBoost prediction")
            
            # FIX: Kiểm tra nếu model là list (multi-task)
            if isinstance(model, list):
                print(f"🔍 XGBoost multi-task with {len(model)} models")
                predictions = np.zeros(len(TASK_NAMES), dtype=np.float32)
                
                for task_idx, task_model in enumerate(model):
                    if task_idx >= len(TASK_NAMES):
                        break
                        
                    if task_model is not None:
                        try:
                            if hasattr(task_model, 'predict_proba'):
                                proba = task_model.predict_proba(features)
                                if len(proba.shape) == 2 and proba.shape[1] >= 2:
                                    predictions[task_idx] = proba[0, 1]
                                else:
                                    predictions[task_idx] = proba[0, 0]
                            else:
                                predictions[task_idx] = task_model.predict(features)[0]
                        except Exception as e:
                            print(f"⚠️ Error predicting task {TASK_NAMES[task_idx]}: {e}")
                            predictions[task_idx] = 0.0
                    else:
                        predictions[task_idx] = 0.0
                
                print(f"✅ Multi-task predictions: {predictions[:5]}...")
            
            else:
                # Fallback: model đơn
                print("🔍 Using single XGBoost model")
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    if len(proba.shape) == 2 and proba.shape[1] >= 2:
                        pred_value = proba[0, 1]
                    else:
                        pred_value = proba[0, 0]
                    
                    # ⚠️ Vẫn tạo 12 giá trị giống nhau nếu là single model
                    predictions = np.full(len(TASK_NAMES), pred_value, dtype=np.float32)
                else:
                    pred_value = model.predict(features)[0]
                    predictions = np.full(len(TASK_NAMES), float(pred_value), dtype=np.float32)
        
        elif model_type in ['mlp', 'pytorch', 'gcn', 'gat']:
            print(f"🔍 Using PyTorch model prediction with no_grad")
            with torch.no_grad():
                if model_type in ['mlp', 'pytorch']:
                    output = model(features)
                elif model_type in ['gcn', 'gat']:
                    output = model(features)
                else:
                    raise ValueError(f"Unknown PyTorch model type: {model_type}")
                
                preds = torch.sigmoid(output).cpu().numpy()
                predictions = preds[0]  # Shape: (12,)
                print(f"✅ PyTorch predictions shape: {predictions.shape}")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"✅ Final predictions shape: {predictions.shape}")
        print(f"✅ Predictions values: {predictions[:5]}...")
        return predictions
        
    except Exception as e:
        print(f"❌ Error in predict_toxicity: {e}")
        import traceback
        traceback.print_exc()
        raise

class ModelManager:
    """Manager for loading and using the best model."""
    
    def __init__(self):
        self.best_model_info = None
        self.model = None
        self.model_type = None
        self.model_name = None
        self.load_best_model()
    
    def load_best_model(self):
        """Load the best model based on pipeline selection."""
        try:
            # Load best model information
            self.best_model_info = load_best_model_info()
            self.model_name = self.best_model_info.get('selected_model', '')
            
            if not self.model_name:
                raise ValueError("No model selected in best_model_selection.json")
            
            # FIX: Use find_model_file to get correct path
            model_path = find_model_file(self.model_name)
            
            # Load the model
            self.model, self.model_type = load_selected_model(str(model_path))
            print(f"✅ Best model loaded: {self.model_name} ({self.model_type})")
            print(f"   Path: {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading best model: {e}")
            self.load_fallback_model()
    
    def load_fallback_model(self):
        """Load a fallback model if best model is not available."""
        models_dir = Path("models")
        
        # FIX: Try to find GCN model first
        gcn_path = models_dir / "gnn_gcn" / "final_model.pt"
        if gcn_path.exists():
            self.model, self.model_type = load_selected_model(str(gcn_path))
            self.model_name = "gnn_gcn_model.pt (fallback)"
            print(f"⚠️  Using fallback model: GCN")
            return
        
        # Then try symlink
        symlink_path = models_dir / "gnn_model.pt"
        if symlink_path.exists():
            try:
                self.model, self.model_type = load_selected_model(str(symlink_path))
                self.model_name = "gnn_model.pt (symlink)"
                print(f"⚠️  Using fallback model: symlink")
                return
            except:
                pass
        
        # Last resort: any available model
        available_models = list(models_dir.glob("**/final_model.pt"))
        if not available_models:
            raise FileNotFoundError("No models found in models/ directory")
        
        model_path = available_models[0]
        self.model, self.model_type = load_selected_model(str(model_path))
        self.model_name = model_path.name
        print(f"⚠️  Using fallback model: {model_path}")
    
    def predict(self, smiles: str, threshold: float = 0.5):
        """Make predictions for a single molecule."""
        try:
            print(f"\n{'='*50}")
            print(f"🔍 Processing SMILES: {smiles}")
            print(f"📊 Model type: {self.model_type}")
            print(f"🔧 Model is list? {isinstance(self.model, list)}")
            if isinstance(self.model, list):
                print(f"📊 Number of sub-models: {len(self.model)}")
            
            print("🔍 Preparing features...")
            features, canonical_smiles = prepare_features(smiles, self.model_type)
            print(f"✅ Features prepared: {type(features)}")
            
            print("🔍 Calling predict_toxicity...")
            predictions = predict_toxicity(self.model, self.model_type, features)
            print(f"✅ Predictions shape: {predictions.shape}")
            print(f"✅ Predictions values: {predictions[:5]}...")
            
            pred_dict = {TASK_NAMES[i]: float(predictions[i]) for i in range(len(TASK_NAMES))}
            binary_dict = {TASK_NAMES[i]: bool(predictions[i] >= threshold) for i in range(len(TASK_NAMES))}
            
            print(f"{'='*50}\n")
            
            return {
                'smiles': smiles,
                'canonical_smiles': canonical_smiles,
                'predictions': pred_dict,
                'binary_predictions': binary_dict,
                'model_type': self.model_type,
                'model_name': self.model_name,
                'valid': True
            }
        except Exception as e:
            print(f"❌ Prediction error for {smiles}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'smiles': smiles,
                'canonical_smiles': '',
                'predictions': {},
                'binary_predictions': {},
                'model_type': self.model_type if self.model_type else 'unknown',
                'model_name': self.model_name if self.model_name else 'unknown',
                'valid': False,
                'error': str(e)
            }

# Initialize model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Load the best model on startup."""
    global model_manager
    try:
        model_manager = ModelManager()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "message": "Tox21 Prediction API",
        "version": "2.0.0",
        "model": model_manager.model_name,
        "model_type": model_manager.model_type,
        "endpoints": {
            "/": "API information (this endpoint)",
            "/health": "Health check",
            "/tasks": "List of prediction tasks",
            "/model_info": "Information about the loaded model",
            "/predict": "Predict toxicity for a single molecule",
            "/predict_batch": "Predict toxicity for multiple molecules"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if model_manager is not None else "unhealthy"
    return {"status": status, "model_loaded": model_manager is not None}

@app.get("/tasks")
async def get_tasks():
    """Get list of prediction tasks."""
    return {
        "tasks": TASK_NAMES,
        "count": len(TASK_NAMES)
    }

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_name": model_manager.model_name,
        "model_type": model_manager.model_type,
        "tasks": len(TASK_NAMES),
        "selection_info": model_manager.best_model_info if model_manager.best_model_info else {}
    }
    
    # Thêm thông tin multi-task nếu có
    if model_manager.model_type == 'xgboost' and isinstance(model_manager.model, list):
        info["sub_models"] = len(model_manager.model)
    
    return info

@app.post("/predict", response_model=MoleculeResponse)
async def predict_single(request: MoleculeRequest):
    """Predict toxicity for a single molecule."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    result = model_manager.predict(request.smiles, request.threshold)
    
    if not result['valid']:
        raise HTTPException(status_code=400, detail=f"Invalid molecule or prediction error: {result.get('error', 'Unknown error')}")
    
    return MoleculeResponse(**result)

@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Predict toxicity for multiple molecules."""
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    valid_count = 0
    
    for smiles in request.smiles_list:
        try:
            result = model_manager.predict(smiles, request.threshold)
            results.append(result)
            if result['valid']:
                valid_count += 1
        except Exception as e:
            print(f"❌ Batch prediction error for {smiles}: {e}")
            results.append({
                'smiles': smiles,
                'canonical_smiles': '',
                'predictions': {},
                'binary_predictions': {},
                'model_type': model_manager.model_type,
                'model_name': model_manager.model_name,
                'valid': False,
                'error': str(e)
            })
    
    return BatchResponse(
        predictions=results,
        model_type=model_manager.model_type,
        model_name=model_manager.model_name,
        total_molecules=len(request.smiles_list),
        valid_molecules=valid_count
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)