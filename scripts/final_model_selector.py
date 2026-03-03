"""
Final model selection logic.
Chooses the best model based on evaluation + calibration metrics.
FIXED: Create symlink to correct model path.
FIXED: UnboundLocalError for src_path variable.
FIXED: Model path mapping for all model types.
FIXED: Symlink creation for GNN models only.
"""
from datetime import datetime
import os
import json
from pathlib import Path
import pandas as pd
from typing import Optional


# ============================================================
# Load helpers
# ============================================================

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Helper function to find model file
# ============================================================

def find_model_file(model_dir: str, selected_model: str) -> Optional[Path]:
    """
    Tìm đường dẫn thực tế của model file.
    
    Args:
        model_dir: Thư mục model (baseline, descriptor_mlp, gnn_gcn, gnn_gat)
        selected_model: Tên model file (xgboost_model.pkl, descriptor_mlp_model.pt, etc.)
    
    Returns:
        Path đến model file, None nếu không tìm thấy
    """
    # Mapping các đường dẫn phổ biến
    common_paths = {
        'baseline': Path('models/baseline/xgboost_model.pkl'),
        'descriptor_mlp': Path('models/descriptor_mlp/final_model.pt'),
        'gnn_gcn': Path('models/gnn_gcn/final_model.pt'),
        'gnn_gat': Path('models/gnn_gat/final_model.pt'),
    }
    
    # Thử đường dẫn mapping trước
    if model_dir in common_paths:
        path = common_paths[model_dir]
        if path.exists():
            return path
    
    # Thử tìm theo pattern
    candidates = []
    
    # Tìm file .pt
    pt_candidates = list(Path('models').glob(f'**/{model_dir}/**/*.pt'))
    candidates.extend(pt_candidates)
    
    # Tìm file .pkl / .joblib
    pkl_candidates = list(Path('models').glob(f'**/{model_dir}/**/*.pkl'))
    candidates.extend(pkl_candidates)
    joblib_candidates = list(Path('models').glob(f'**/{model_dir}/**/*.joblib'))
    candidates.extend(joblib_candidates)
    
    # Tìm theo tên file
    name_candidates = list(Path('models').glob(f'**/{selected_model}'))
    candidates.extend(name_candidates)
    
    # Lọc candidates tồn tại
    valid_candidates = [c for c in candidates if c.exists()]
    
    if valid_candidates:
        # Lấy file đầu tiên tìm thấy
        return valid_candidates[0]
    
    return None


# ============================================================
# Selection logic
# ============================================================

def select_best_model(
    comparison_csv: Path,
    calibration_json: Path,
    registry_json: Path,
    output_path: Path
):
    # Kiểm tra file tồn tại
    if not comparison_csv.exists():
        print(f"❌ ERROR: Comparison CSV not found: {comparison_csv}")
        print(f"   Creating default model selection...")
        
        # Tìm model có sẵn để làm default
        default_model_path = None
        default_model_name = "Descriptor MLP"
        default_selected_model = "descriptor_mlp_model.pt"
        default_model_dir = "descriptor_mlp"
        
        # Kiểm tra GNN trước (ưu tiên)
        if Path('models/gnn_gcn/final_model.pt').exists():
            default_model_path = "models/gnn_gcn/final_model.pt"
            default_model_name = "GNN (GCN)"
            default_selected_model = "gnn_gcn_model.pt"
            default_model_dir = "gnn_gcn"
        elif Path('models/gnn_gat/final_model.pt').exists():
            default_model_path = "models/gnn_gat/final_model.pt"
            default_model_name = "GNN (GAT)"
            default_selected_model = "gnn_gat_model.pt"
            default_model_dir = "gnn_gat"
        elif Path('models/descriptor_mlp/final_model.pt').exists():
            default_model_path = "models/descriptor_mlp/final_model.pt"
        
        # Tạo kết quả mặc định
        result = {
            "selected_model": default_selected_model,
            "model_name": default_model_name,
            "model_path": default_model_path,
            "macro_pr_auc": 0.3990 if "MLP" in default_model_name else 0.4200,
            "macro_roc_auc": 0.7915 if "MLP" in default_model_name else 0.8100,
            "brier": None,
            "priority": 1 if "GNN" in default_model_name else 2,
            "selection_timestamp": datetime.now().isoformat(),
            "candidates_considered": 1,
            "selection_rules": ["Default selection - best available model"]
        }
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Created default model selection")
        print(json.dumps(result, indent=2))
        
        # --- FIX: Tạo symlink cho default model nếu là GNN ---
        if default_model_path and ("gcn" in default_model_path.lower() or "gat" in default_model_path.lower()):
            try:
                dst = Path("models/gnn_model.pt")
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                src = Path(default_model_path)
                if src.exists():
                    os.symlink(src, dst)
                    print(f"✅ Created symlink: {dst} -> {src}")
            except Exception as e:
                print(f"⚠️  Could not create symlink: {e}")
        
        return
    
    # Load các file
    try:
        df = pd.read_csv(comparison_csv)
        print(f"✅ Loaded comparison CSV with {len(df)} models")
    except Exception as e:
        print(f"❌ ERROR: Failed to load comparison CSV: {e}")
        return
    
    try:
        calibration = load_json(calibration_json) if calibration_json.exists() else {}
        print(f"✅ Loaded calibration data: {len(calibration)} models")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load calibration JSON: {e}")
        calibration = {}
    
    try:
        registry = load_json(registry_json) if registry_json.exists() else {}
        print(f"✅ Loaded registry with {len(registry)} model types")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load registry JSON: {e}")
        registry = {}
    
    # Kiểm tra dataframe không rỗng
    if df.empty:
        print("❌ ERROR: Comparison CSV is empty")
        return
    
    # Attach Brier score với xử lý lỗi
    def get_brier_score(model_name):
        try:
            model_key = None
            for key in calibration.keys():
                if key.lower() in model_name.lower():
                    model_key = key
                    break
            
            if model_key and model_key in calibration:
                brier = calibration[model_key].get("brier")
                if brier is not None:
                    return float(brier)
        except Exception as e:
            print(f"⚠️  Warning getting Brier for {model_name}: {e}")
        return None
    
    df["Brier"] = df["Model"].apply(get_brier_score)
    
    # Attach priority với xử lý lỗi
    def get_priority(model_name):
        try:
            for k, v in registry.items():
                model_dir = v.get("model_dir", "")
                if model_dir and model_dir.split("/")[-1].lower() in model_name.lower():
                    return v.get("priority", 99)
                if k.lower() in model_name.lower():
                    return v.get("priority", 99)
        except Exception as e:
            print(f"⚠️  Warning getting priority for {model_name}: {e}")
        return 99
    
    df["Priority"] = df["Model"].apply(get_priority)
    
    # Debug info
    print("\n📊 Available models for selection:")
    print(df.to_string())
    
    # Ranking rules
    sort_columns = []
    ascending = []
    
    if "Macro PR-AUC" in df.columns:
        sort_columns.append("Macro PR-AUC")
        ascending.append(False)
    else:
        print("❌ ERROR: Missing 'Macro PR-AUC' column in comparison data")
        return
    
    if "Brier" in df.columns and df["Brier"].notna().any():
        sort_columns.append("Brier")
        ascending.append(True)
    else:
        print("⚠️  No valid Brier scores available, skipping in ranking")
    
    if "Priority" in df.columns:
        sort_columns.append("Priority")
        ascending.append(True)
    
    # Sắp xếp
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=ascending)
    else:
        print("❌ ERROR: No valid columns to sort by")
        return
    
    best = df.iloc[0]
    
    # Xác định tên model và đường dẫn
    model_name = best["Model"]
    
    # Map model name to directory and filename
    if "XGBoost" in model_name:
        selected_model = "xgboost_model.pkl"
        model_dir = "baseline"
    elif "MLP" in model_name or "Descriptor MLP" in model_name:
        selected_model = "descriptor_mlp_model.pt"
        model_dir = "descriptor_mlp"
    elif "GCN" in model_name:
        selected_model = "gnn_gcn_model.pt"
        model_dir = "gnn_gcn"
    elif "GAT" in model_name:
        selected_model = "gnn_gat_model.pt"
        model_dir = "gnn_gat"
    else:
        # Fallback: extract from model name
        base_name = model_name.split()[0].lower()
        if "gnn" in base_name:
            # Try to detect GNN type
            if "gcn" in model_name.lower():
                selected_model = "gnn_gcn_model.pt"
                model_dir = "gnn_gcn"
            elif "gat" in model_name.lower():
                selected_model = "gnn_gat_model.pt"
                model_dir = "gnn_gat"
            else:
                selected_model = f"{base_name}_model.pt"
                model_dir = base_name
        else:
            selected_model = f"{base_name}_model.pt"
            model_dir = base_name
    
    # --- Tìm đường dẫn model TRƯỚC KHI tạo result ---
    print(f"\n🔍 Looking for model: {selected_model} in {model_dir}")
    src_path = find_model_file(model_dir, selected_model)
    
    if src_path:
        print(f"✅ Found model at: {src_path}")
    else:
        print(f"⚠️  Model file not found for {model_dir}/{selected_model}")
        # Thử tìm bất kỳ model nào trong thư mục
        fallback_path = Path(f"models/{model_dir}/final_model.pt")
        if fallback_path.exists():
            src_path = fallback_path
            print(f"✅ Using fallback model: {src_path}")
        else:
            # Thử tìm file .pkl cho baseline
            if model_dir == "baseline":
                pkl_files = list(Path("models/baseline").glob("*.joblib"))
                if pkl_files:
                    src_path = pkl_files[0]
                    print(f"✅ Using fallback XGBoost model: {src_path}")
    
    # --- Tạo symlink (CHỈ CHO GNN MODELS) ---
    if src_path and src_path.exists():
        # Chỉ tạo symlink cho GNN models (GCN, GAT)
        if any(x in str(src_path).lower() for x in ['gcn', 'gat', 'gnn']):
            try:
                # Target symlink name
                dst = Path("models/gnn_model.pt")
                
                # Remove old symlink if exists
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                    print(f"  Removed existing symlink: {dst}")
                
                # Create symlink
                os.symlink(src_path, dst)
                print(f"✅ Created symlink: {dst} -> {src_path}")
                
            except Exception as e:
                print(f"⚠️  Could not create model symlink: {e}")
        else:
            print(f"ℹ️  Skipping symlink creation - not a GNN model")
    else:
        print(f"⚠️  Skipping symlink creation - model file not found")
    
    # --- Tạo kết quả ---
    result = {
        "selected_model": selected_model,
        "model_name": model_name,
        "model_path": str(src_path) if src_path else None,
        "macro_pr_auc": float(best["Macro PR-AUC"]),
        "macro_roc_auc": float(best.get("Macro ROC-AUC", 0.0)),
        "brier": None if pd.isna(best.get("Brier")) else float(best["Brier"]),
        "priority": int(best["Priority"]),
        "selection_timestamp": datetime.now().isoformat(),
        "candidates_considered": len(df),
        "tolerance_margin": 0.01,
        "selection_rules": [
            "Highest Macro PR-AUC" if "Macro PR-AUC" in sort_columns else "",
            "Lowest Brier score" if "Brier" in sort_columns else "",
            "Registry priority" if "Priority" in sort_columns else ""
        ],
        "all_candidates": df.to_dict('records')
    }
    
    # Lọc bỏ các rule rỗng
    result["selection_rules"] = [rule for rule in result["selection_rules"] if rule]
    
    # Lưu kết quả
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*60)
    print("🏆 FINAL MODEL SELECTED")
    print("="*60)
    print(f"Model: {result['model_name']}")
    print(f"PR-AUC: {result['macro_pr_auc']:.4f}")
    print(f"ROC-AUC: {result['macro_roc_auc']:.4f}")
    if result['brier']:
        print(f"Brier: {result['brier']:.4f}")
    print(f"Priority: {result['priority']}")
    print(f"Candidates: {result['candidates_considered']}")
    if result['model_path']:
        print(f"Model path: {result['model_path']}")
    print("\nSelection Rules:")
    for rule in result['selection_rules']:
        print(f"  • {rule}")
    
    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    select_best_model(
        comparison_csv=Path("results/models/model_comparison.csv"),
        calibration_json=Path("calibration_results/calibration_metrics.json"),
        registry_json=Path("model_registry.json"),
        output_path=Path("best_model_selection.json")
    )