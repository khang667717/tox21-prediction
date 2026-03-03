#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        TOX21 POST-PIPELINE ANALYSIS & DEPLOYMENT             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 Start time: $(date)"
echo ""
echo "Options:"
echo "  --all              Run all post-pipeline steps (default)"
echo "  --demo             Run demo predictions only"
echo "  --interpret        Run interpretability analysis only"
echo "  --api              Start FastAPI server only"
echo "  --test             Run data integrity tests only"
echo "  --custom           Custom selection of steps"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================
RUN_ALL=true
RUN_DEMO=false
RUN_INTERPRET=false
RUN_API=false
RUN_TEST=false

# Process arguments
for arg in "$@"
do
    case $arg in
        --all)
        RUN_ALL=true
        shift
        ;;
        --demo)
        RUN_DEMO=true
        RUN_ALL=false
        shift
        ;;
        --interpret)
        RUN_INTERPRET=true
        RUN_ALL=false
        shift
        ;;
        --api)
        RUN_API=true
        RUN_ALL=false
        shift
        ;;
        --test)
        RUN_TEST=true
        RUN_ALL=false
        shift
        ;;
        --custom)
        RUN_ALL=false
        shift
        ;;
    esac
done

# If no specific flags and not --all, set to --all
if [ "$RUN_ALL" = false ] && \
   [ "$RUN_DEMO" = false ] && \
   [ "$RUN_INTERPRET" = false ] && \
   [ "$RUN_API" = false ] && \
   [ "$RUN_TEST" = false ]; then
    RUN_ALL=true
fi

echo "📊 Configuration:"
echo "   Run All: $RUN_ALL"
echo "   Run Demo: $RUN_DEMO"
echo "   Run Interpretability: $RUN_INTERPRET"
echo "   Run API: $RUN_API"
echo "   Run Tests: $RUN_TEST"
echo ""

# ============================================================================
# BƯỚC 1: KIỂM TRA PIPELINE ĐÃ CHẠY THÀNH CÔNG
# ============================================================================
echo "🔍 STEP 1: Checking if pipeline completed successfully..."
echo ""

# Check if best model selection exists
if [ ! -f "best_model_selection.json" ]; then
    echo "❌ ERROR: best_model_selection.json not found!"
    echo "   Please run the full pipeline first: ./run_full_pipeline.sh"
    exit 1
fi

# Check if selected model exists
SELECTED_MODEL=$(python -c "
import json, os
with open('best_model_selection.json', 'r') as f:
    data = json.load(f)
model = data.get('selected_model', '')
# Extract just the filename
print(os.path.basename(model))
")

echo "   Looking for model: $SELECTED_MODEL"

# Find the model file directly
# Find the model file directly - FIXED for XGBoost
MODEL_PATH=""

# Đọc đường dẫn thực tế từ best_model_selection.json
MODEL_PATH=$(python -c "
import json
with open('best_model_selection.json') as f:
    data = json.load(f)
print(data.get('model_path', ''))
")

# Nếu không có, thử tìm theo tên
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "   Path from JSON not found, searching..."
    
    if [ "$SELECTED_MODEL" = "xgboost_model.pkl" ]; then
        # Tìm file XGBoost joblib
        MODEL_PATH=$(find models/baseline -name "*.joblib" 2>/dev/null | head -n 1)
    elif [ "$SELECTED_MODEL" = "gnn_gcn_model.pt" ]; then
        MODEL_PATH="models/gnn_gcn/final_model.pt"
    elif [ "$SELECTED_MODEL" = "gnn_gat_model.pt" ]; then
        MODEL_PATH="models/gnn_gat/final_model.pt"
    elif [ "$SELECTED_MODEL" = "descriptor_mlp_model.pt" ]; then
        MODEL_PATH="models/descriptor_mlp/final_model.pt"
    fi
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Selected model '$SELECTED_MODEL' not found!"
    echo "   Please check models directory"
    exit 1
fi

echo "✅ Pipeline check passed"
echo "   Selected model: $SELECTED_MODEL"
echo "   Model path: $MODEL_PATH"
echo ""

# ============ FIX: Đọc metrics đúng cách ============
MODEL_INFO=$(python -c "
import json
with open('best_model_selection.json', 'r') as f:
    data = json.load(f)
# FIX: Đọc trực tiếp từ root object, không qua 'model_metrics'
pr_auc = data.get('macro_pr_auc', 0.0)
roc_auc = data.get('macro_roc_auc', 0.0)
print(f'Macro PR-AUC: {pr_auc:.4f}')
print(f'Macro ROC-AUC: {roc_auc:.4f}')
")

echo "📈 Model Performance:"
echo "$MODEL_INFO"
echo ""

# ============================================================================
# BƯỚC 2: CHẠY DEMO PREDICTIONS - FIXED
# ============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_DEMO" = true ]; then
    echo "🎯 STEP 2: Running Demo Predictions"
    echo "   Script: demo_predict.py"
    echo "   Output: demo_results/"
    echo ""
    
    # Create demo results directory
    mkdir -p demo_results
    
    echo "   Running demo predictions with selected model..."
    
    # FIX: Chạy demo_predict.py ở chế độ non-interactive bằng echo pipe
    echo -e "6\n0" | python demo_predict.py 2>/dev/null || \
    echo "   ⚠️  Demo prediction might have issues (try running manually: python demo_predict.py)"
    
    # Check if demo results were created
    if [ -d "demo_results" ] && [ "$(ls -A demo_results 2>/dev/null)" ]; then
        echo ""
        echo "   ✅ Demo results created in demo_results/:"
        ls -la demo_results/
    else
        echo "   ⚠️  No demo results created yet"
    fi
    
    echo "✅ Demo predictions completed"
    echo ""
fi

# ============================================================================
# BƯỚC 3: INTERPRETABILITY ANALYSIS
# ============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_INTERPRET" = true ]; then
    echo "🔬 STEP 3: Interpretability Analysis"
    echo "   Script: interpretability.py"
    echo "   Output: interpretability/"
    echo ""
    
    # Create interpretability directory
    mkdir -p interpretability
    
    # Determine model type and directory correctly based on actual paths
    if echo "$SELECTED_MODEL" | grep -qi "xgboost"; then
        MODEL_TYPE="xgboost"
        MODEL_DIR="models/baseline"
    elif echo "$SELECTED_MODEL" | grep -qi "mlp"; then
        MODEL_TYPE="mlp"
        MODEL_DIR="models/descriptor_mlp"
    elif echo "$SELECTED_MODEL" | grep -qi "gcn"; then
        MODEL_TYPE="gnn"
        MODEL_DIR="models/gnn_gcn"
    elif echo "$SELECTED_MODEL" | grep -qi "gat"; then
        MODEL_TYPE="gnn"
        MODEL_DIR="models/gnn_gat"
    else
        MODEL_TYPE="all"
    fi
    
    if [ "$MODEL_TYPE" = "all" ]; then
        echo "   Analyzing all available models..."
        echo ""
        
        # Analyze XGBoost if exists
        if [ -d "models/baseline" ]; then
            echo "   📊 Analyzing XGBoost model..."
            python interpretability.py \
                --model_type xgboost \
                --model_dir models/baseline \
                --output_dir interpretability/xgboost \
                --n_tasks 3 || \
            echo "   ⚠️  XGBoost interpretability might have issues"
        fi
        
        # Analyze MLP if exists
        if [ -d "models/descriptor_mlp" ]; then
            echo "   📊 Analyzing MLP model..."
            python interpretability.py \
                --model_type mlp \
                --model_dir models/descriptor_mlp \
                --output_dir interpretability/mlp \
                --n_tasks 3 || \
            echo "   ⚠️  MLP interpretability might have issues"
        fi
        
        # Analyze GCN if exists
        if [ -d "models/gnn_gcn" ]; then
            echo "   📊 Analyzing GCN model..."
            python interpretability.py \
                --model_type gnn \
                --model_dir models/gnn_gcn \
                --output_dir interpretability/gcn \
                --n_tasks 2 || \
            echo "   ⚠️  GCN interpretability might have issues"
        fi
        
        # Analyze GAT if exists
        if [ -d "models/gnn_gat" ]; then
            echo "   📊 Analyzing GAT model..."
            python interpretability.py \
                --model_type gnn \
                --model_dir models/gnn_gat \
                --output_dir interpretability/gat \
                --n_tasks 2 || \
            echo "   ⚠️  GAT interpretability might have issues"
        fi
    else
        echo "   📊 Analyzing $MODEL_TYPE model..."
        python interpretability.py \
            --model_type $MODEL_TYPE \
            --model_dir $MODEL_DIR \
            --output_dir interpretability \
            --n_tasks 3 || \
        echo "   ⚠️  Interpretability analysis might have issues"
    fi
    
    # FIX: Sửa lệnh head để hiển thị kết quả đúng cách
    if [ -d "interpretability" ] && [ "$(ls -A interpretability 2>/dev/null)" ]; then
        echo ""
        echo "   ✅ Interpretability results created in interpretability/:"
        # Cách 1: Dùng head -n (có -n)
        find interpretability -name "*.png" -o -name "*.json" 2>/dev/null | /usr/bin/head -10

        # ls -la interpretability/*/*.png 2>/dev/null | head -10
    fi
    
    echo "✅ Interpretability analysis completed"
    echo ""
fi

# ============================================================================
# BƯỚC 4: DATA INTEGRITY TESTS
# ============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_TEST" = true ]; then
    echo "🧪 STEP 4: Data Integrity Tests"
    echo "   Script: test_split_integrity.py"
    echo ""
    
    echo "   Running comprehensive data integrity tests..."
    python test_split_integrity.py
    
    if [ $? -eq 0 ]; then
        echo "✅ All data integrity tests passed"
    else
        echo "⚠️  Some tests failed or were skipped"
    fi
    echo ""
fi

# ============================================================================
# BƯỚC 5: API DEPLOYMENT
# ============================================================================
if [ "$RUN_ALL" = true ] || [ "$RUN_API" = true ]; then
    echo "🚀 STEP 5: API Deployment"
    echo "   Script: FastAPI_fixed.py"
    echo "   URL: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    echo ""
    
    echo "   Starting FastAPI server in background..."
    echo "   (Press Ctrl+C to stop the server)"
    echo ""
    
    # Check if port 8000 is already in use
    PORT_TO_USE=8000
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "⚠️  Port 8000 is already in use"
            for PORT in {8001..8010}; do
                if ! lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
                    echo "   Using port $PORT instead"
                    PORT_TO_USE=$PORT
                    break
                fi
            done
        fi
    fi
    
    echo ""
    echo "📡 Starting FastAPI server on port $PORT_TO_USE..."
    echo "   API will be available at: http://localhost:$PORT_TO_USE"
    echo "   Interactive docs at: http://localhost:$PORT_TO_USE/docs"
    echo ""
    
    # Start the server in background
    python FastAPI_fixed.py &
    SERVER_PID=$!
    
    # Wait a bit for server to start
    sleep 5
    
    # Check if server is running
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "✅ FastAPI server started successfully (PID: $SERVER_PID)"
        echo ""
        
        # Display available endpoints
        echo "📋 Available endpoints:"
        echo "   GET  /              - API information"
        echo "   GET  /health        - Health check"
        echo "   GET  /tasks         - List of prediction tasks"
        echo "   GET  /model_info    - Information about loaded model"
        echo "   POST /predict       - Predict toxicity for single molecule"
        echo "   POST /predict_batch - Predict toxicity for multiple molecules"
        echo ""
        
        # FIX: Test health endpoint - không dùng json.tool nếu không có
        echo "🧪 Testing health endpoint..."
        curl -s "http://localhost:$PORT_TO_USE/health" | python -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || \
        curl -s "http://localhost:$PORT_TO_USE/health" || \
        echo "   ⚠️  Health check might be unavailable"
        echo ""
        
        # FIX: Test tasks endpoint - in raw nếu json.tool lỗi
        echo "🧪 Testing tasks endpoint..."
        curl -s "http://localhost:$PORT_TO_USE/tasks" | python -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || \
        curl -s "http://localhost:$PORT_TO_USE/tasks" || \
        echo "   ⚠️  Tasks endpoint might be unavailable"
        echo ""
        
        echo "💡 Example API calls:"
        echo ""
        echo "   # Single prediction"
        echo '   curl -X POST "http://localhost:'$PORT_TO_USE'/predict" \'
        echo '        -H "Content-Type: application/json" \'
        echo '        -d '\''{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "threshold": 0.5}'\'''
        echo ""
        echo '   # Batch prediction'
        echo '   curl -X POST "http://localhost:'$PORT_TO_USE'/predict_batch" \'
        echo '        -H "Content-Type: application/json" \'
        echo '        -d '\''{"smiles_list": ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"], "threshold": 0.5}'\'''
        echo ""
        
        # Check if user wants to keep server running
        read -p "❓ Keep server running? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "⏳ Server will continue running. Press Ctrl+C to stop."
            echo "   Use 'kill $SERVER_PID' to stop it later."
            wait $SERVER_PID
        else
            echo "🛑 Stopping server..."
            kill $SERVER_PID 2>/dev/null
            wait $SERVER_PID 2>/dev/null
            echo "✅ Server stopped"
        fi
    else
        echo "❌ Failed to start FastAPI server"
        echo "   Check if all dependencies are installed:"
        echo "   pip install fastapi uvicorn"
    fi
    echo ""
fi

# ============================================================================
# TỔNG KẾT - FIXED
# ============================================================================
echo "📊 POST-PIPELINE SUMMARY"
echo "══════════════════════════════════════════════════════════════"

# Show what was created
echo ""
echo "📁 Generated Directories & Files:"
echo ""

if [ -d "demo_results" ] && [ "$(ls -A demo_results 2>/dev/null)" ]; then
    echo "   ├── 📂 demo_results/ - Demo predictions & visualizations"
    # FIX: Sử dụng ls -1 | head -5 thay vì find | head
    ls -1 demo_results/ 2>/dev/null | head -5 | while read file; do
        echo "   │   ├── $file"
    done
    if [ $(ls -1 demo_results/ 2>/dev/null | wc -l) -gt 5 ]; then
        echo "   │   └── ... and more"
    fi
else
    echo "   ├── 📂 demo_results/ - No demo results yet"
fi

if [ -d "interpretability" ] && [ "$(ls -A interpretability 2>/dev/null)" ]; then
    echo "   ├── 📂 interpretability/ - Interpretability analysis"
    # FIX: Sử dụng find đúng syntax
    find interpretability -name "*.png" 2>/dev/null | head -3 | while read file; do
        echo "   │   ├── $(basename $file)"
    done
    if [ $(find interpretability -name "*.png" 2>/dev/null | wc -l) -gt 3 ]; then
        echo "   │   └── ... and more"
    fi
fi

if [ -f "best_model_selection.json" ]; then
    echo "   ├── 📄 best_model_selection.json - Selected model info"
fi

# Show next steps
echo ""
echo "🚀 Next Steps:"
echo ""
echo "   1. Run demo predictions interactively:"
echo "      python demo_predict.py"
echo ""
echo "   2. Start API server:"
echo "      python FastAPI_fixed.py"
echo ""
echo "   3. Test API with curl (if server is running):"
echo '      curl -X POST "http://localhost:8000/predict" \'
echo '           -H "Content-Type: application/json" \'
echo '           -d '\''{"smiles": "CCO", "threshold": 0.5}'\'''
echo ""

echo "⏱️  End time: $(date)"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              POST-PIPELINE COMPLETED! ✅                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
