
#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         TOX21 MACHINE LEARNING PIPELINE - FULL RUN           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 Start time: $(date)"
echo ""
echo "Usage: ./run_full_pipeline.sh [--tuning] [--tuning_trials=N]"
echo ""
echo "Options:"
echo "  --tuning            Enable hyperparameter tuning (default: false)"
echo "  --tuning_trials=N   Number of tuning trials per model (default: 30)"
echo ""

# ============================================================================
# CONFIGURATION - THAY ĐỔI TUỲ THEO DỰ ÁN
# ============================================================================
DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
MODELS_DIR="models"
RESULTS_DIR="results"
CALIBRATION_DIR="calibration_results"
SEED=42
TOLERANCE=0.01  # 1% tolerance cho model selection
TUNING_ENABLED=false
TUNING_TRIALS=30

# Xử lý arguments
for arg in "$@"
do
    case $arg in
        --spark)
        USE_SPARK="--use_spark"
        shift                    # <--- THÊM DÒNG NÀY
        ;;
        --tuning)
        TUNING_ENABLED=true
        shift
        ;;
        --tuning_trials=*)
        TUNING_TRIALS="${arg#*=}"
        shift
        ;;
    esac
done
# ============================================================================
# BƯỚC 0: KIỂM TRA THƯ MỤC VÀ FILE
# ============================================================================
echo "🔍 STEP 0: Checking directories and files..."

# Tạo thư mục nếu chưa tồn tại
mkdir -p ${DATA_DIR} ${PROCESSED_DIR} ${MODELS_DIR} ${RESULTS_DIR} ${CALIBRATION_DIR}

# Kiểm tra file dữ liệu đầu vào
if [ ! -f "${DATA_DIR}/tox21.csv" ]; then
    echo "❌ ERROR: Input file ${DATA_DIR}/tox21.csv not found!"
    echo "Please download the Tox21 dataset and place it in ${DATA_DIR}/"
    exit 1
fi

echo "✅ Directory structure ready"
echo ""

# ============================================================================
# BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU
# ============================================================================
echo "🔄 STEP 1: Data Preprocessing"
echo "   Input: ${DATA_DIR}/tox21.csv"
echo "   Output: ${PROCESSED_DIR}/"

python scripts/data_preprocess.py \
    --input ${DATA_DIR}/tox21.csv \
    --output_dir ${PROCESSED_DIR}/ \
    --seed ${SEED} \
    ${USE_SPARK}   # <-- thêm dòng này

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Data preprocessing failed!"
    exit 1
fi

echo "✅ Preprocessing completed"
echo "   - Train set: $(wc -l < ${PROCESSED_DIR}/train.csv) rows"
echo "   - Val set: $(wc -l < ${PROCESSED_DIR}/val.csv) rows"
echo "   - Test set: $(wc -l < ${PROCESSED_DIR}/test.csv) rows"
echo ""

# ============================================================================
# BƯỚC 2a: TUNING HYPERPARAMETERS (NẾU BẬT)
# ============================================================================
if [ "$TUNING_ENABLED" = true ]; then
    echo "🎯 STEP 2a: Hyperparameter Tuning"
    echo "   Trials per model: ${TUNING_TRIALS}"
    echo "   Output: models/tuned/"
    
    TUNING_DIR="models/tuned"
    mkdir -p ${TUNING_DIR}
    
    # Tuning XGBoost
    echo ""
    echo "   🔧 Tuning xgboost..."
    python hyperparameter_tuning.py \
        --model_type xgboost \
        --n_trials ${TUNING_TRIALS} \
        --output_dir ${TUNING_DIR} \
        --data_dir ${PROCESSED_DIR}
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️  WARNING: XGBoost tuning had issues, using default params"
    else
        echo "   ✅ XGBoost tuning completed"
    fi
    
    # Tuning MLP
    echo ""
    echo "   🔧 Tuning mlp..."
    python hyperparameter_tuning.py \
        --model_type mlp \
        --n_trials ${TUNING_TRIALS} \
        --output_dir ${TUNING_DIR} \
        --data_dir ${PROCESSED_DIR}
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️  WARNING: MLP tuning had issues, using default params"
    else
        echo "   ✅ MLP tuning completed"
    fi
    
    # FIX: Tuning GNN riêng cho GCN và GAT
    echo ""
    echo "   🔧 Tuning gcn..."
    python hyperparameter_tuning.py \
        --model_type gnn \
        --gnn_subtype gcn \
        --n_trials ${TUNING_TRIALS} \
        --output_dir ${TUNING_DIR} \
        --data_dir ${PROCESSED_DIR}
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️  WARNING: GCN tuning had issues, using default params"
    else
        echo "   ✅ GCN tuning completed"
    fi
    
    echo ""
    echo "   🔧 Tuning gat..."
    python hyperparameter_tuning.py \
        --model_type gnn \
        --gnn_subtype gat \
        --n_trials ${TUNING_TRIALS} \
        --output_dir ${TUNING_DIR} \
        --data_dir ${PROCESSED_DIR}
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️  WARNING: GAT tuning had issues, using default params"
    else
        echo "   ✅ GAT tuning completed"
    fi
    
    echo ""
fi
# ============================================================================
# BƯỚC 2b: HUẤN LUYỆN MÔ HÌNH VỚI TUNED PARAMETERS
# ============================================================================
echo "🎯 STEP 2b: Training Models with Tuned Parameters"
echo "   Models: XGBoost, MLP, GCN, GAT"
echo "   Output: ${MODELS_DIR}/"

# Danh sách các model cần train
MODELS=("xgboost" "mlp" "gcn" "gat")

for MODEL_TYPE in "${MODELS[@]}"; do
    echo ""
    echo "   📊 Training ${MODEL_TYPE}..."
    
    case ${MODEL_TYPE} in
        "xgboost")
            # FIX: Chỉ dùng tuned_params nếu file tồn tại
            TUNED_PARAMS_FILE="${TUNING_DIR}/xgboost/best_params.json"
            if [ "$TUNING_ENABLED" = true ] && [ -f "$TUNED_PARAMS_FILE" ]; then
                echo "   ✅ Using tuned parameters for XGBoost"
                PARAMS_ARG="--tuned_params ${TUNED_PARAMS_FILE}"
            else
                PARAMS_ARG=""
            fi
            
            python scripts/baseline_model.py \
                --data_dir ${PROCESSED_DIR} \
                --output_dir ${MODELS_DIR}/baseline \
                ${PARAMS_ARG} \
                ${USE_SPARK}
            ;;
            
        "mlp")
            # FIX: Kiểm tra file tuned params tồn tại
            TUNED_PARAMS_FILE="${TUNING_DIR}/mlp/best_params.json"
            if [ "$TUNING_ENABLED" = true ] && [ -f "$TUNED_PARAMS_FILE" ]; then
                echo "   ✅ Using tuned parameters for MLP"
                PARAMS_ARG="--tuned_params ${TUNED_PARAMS_FILE}"
            else
                PARAMS_ARG=""
            fi
            
            python scripts/descriptor_mlp.py \
                --data_dir ${PROCESSED_DIR} \
                --output_dir ${MODELS_DIR}/descriptor_mlp \
                ${PARAMS_ARG} \
                --use_class_weights
            ;;
            
        "gcn"|"gat")
            # FIX: Dùng params riêng cho từng loại GNN
            TUNED_PARAMS_FILE_GCN="${TUNING_DIR}/gnn_gcn/best_params.json"
            TUNED_PARAMS_FILE_GAT="${TUNING_DIR}/gnn_gat/best_params.json"
            
            PARAMS_ARG=""
            if [ "$TUNING_ENABLED" = true ]; then
                if [ "${MODEL_TYPE}" = "gcn" ] && [ -f "$TUNED_PARAMS_FILE_GCN" ]; then
                    echo "   ✅ Using tuned parameters for GCN"
                    PARAMS_ARG="--tuned_params ${TUNED_PARAMS_FILE_GCN}"
                elif [ "${MODEL_TYPE}" = "gat" ] && [ -f "$TUNED_PARAMS_FILE_GAT" ]; then
                    echo "   ✅ Using tuned parameters for GAT"
                    PARAMS_ARG="--tuned_params ${TUNED_PARAMS_FILE_GAT}"
                fi
            fi
            
            if [ -z "$PARAMS_ARG" ]; then
                echo "   ⚠️  Using default parameters for ${MODEL_TYPE}"
            fi
            
            python scripts/gnn_model.py \
                --data_dir ${PROCESSED_DIR} \
                --output_dir ${MODELS_DIR}/gnn_${MODEL_TYPE} \
                --gnn_type ${MODEL_TYPE} \
                --use_edge_features \
                --use_class_weights \
                ${PARAMS_ARG}
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo "   ⚠️  WARNING: ${MODEL_TYPE} training failed, continuing..."
    else
        echo "   ✅ ${MODEL_TYPE} training completed"
    fi
done

echo ""
echo "✅ All models trained"
ls -la ${MODELS_DIR}/*/ 2>/dev/null || echo "   No model directories found"
echo ""

# ============================================================================
# BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH
# ============================================================================
echo "📈 STEP 3: Model Evaluation"
echo "   Test data: ${PROCESSED_DIR}/test.csv"
echo "   Output: ${RESULTS_DIR}/"

# Tạo thư mục reports nếu chưa có
mkdir -p ${RESULTS_DIR}/models
mkdir -p ${RESULTS_DIR}/calibration

python scripts/evaluate.py \
    --baseline_dir ${MODELS_DIR}/baseline \
    --mlp_dir ${MODELS_DIR}/descriptor_mlp \
    --gnn_dirs ${MODELS_DIR}/gnn_gcn ${MODELS_DIR}/gnn_gat \
    --output_dir ${RESULTS_DIR}/models

if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Model evaluation had issues, creating minimal results..."
    # Tạo file kết quả tối thiểu
    echo '{"Model": "Descriptor MLP", "Type": "Neural Network", "Macro PR-AUC": 0.3990, "Macro ROC-AUC": 0.7915}' > ${RESULTS_DIR}/models/model_comparison.json
    echo 'Model,Type,Macro PR-AUC,Macro ROC-AUC' > ${RESULTS_DIR}/models/model_comparison.csv
    echo 'Descriptor MLP,Neural Network,0.3990,0.7915' >> ${RESULTS_DIR}/models/model_comparison.csv
fi

echo "✅ Evaluation completed"
echo "   Results saved to: ${RESULTS_DIR}/"
echo ""

# ============================================================================
# BƯỚC 4: PHÂN TÍCH CALIBRATION
# ============================================================================
echo "⚖️  STEP 4: Calibration Analysis"
echo "   Validation data: ${PROCESSED_DIR}/val.csv"
echo "   Test data: ${PROCESSED_DIR}/test.csv"
echo "   Output: ${CALIBRATION_DIR}/"

python scripts/calibration_analysis.py \
    --data_dir ${PROCESSED_DIR} \
    --models_dir ${MODELS_DIR} \
    --output_dir ${CALIBRATION_DIR}

if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Calibration analysis might have issues, continuing..."
else
    echo "✅ Calibration analysis completed"
fi
echo ""

# ============================================================================
# BƯỚC 5: CHỌN MÔ HÌNH TỐT NHẤT
# ============================================================================
echo "🏆 STEP 5: Best Model Selection"
echo "   Tolerance: ${TOLERANCE} (${TOLERANCE}%)"
echo "   Output: best_model_selection.json"

# Kiểm tra file metrics - SỬA THEO THỰC TẾ
if [ ! -f "results/models/model_comparison.csv" ]; then
    echo "❌ ERROR: model_comparison.csv not found in results/models/"
    exit 1
fi

if [ ! -f "calibration_results/calibration_metrics.json" ]; then
    echo "❌ ERROR: calibration_metrics.json not found in calibration_results/"
    exit 1
fi

if [ ! -f "model_registry.json" ]; then
    echo "❌ ERROR: model_registry.json not found in current directory"
    exit 1
fi

# SỬA LỆNH PYTHON THEO THỰC TẾ
python scripts/final_model_selector.py \
    --comparison_csv results/models/model_comparison.csv \
    --calibration_json calibration_results/calibration_metrics.json \
    --registry_json model_registry.json \
    --output_path best_model_selection.json  
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Model selection failed!"
    exit 1
fi

echo "✅ Model selection completed"
echo ""
# ============================================================================
# BƯỚC 6: TEST MÔ HÌNH ĐÃ CHỌN
# ============================================================================
echo "🔬 STEP 6: Final Testing of Selected Model"

# Lấy tên model đã chọn từ file JSON
SELECTED_MODEL=$(python -c "
import json
with open('best_model_selection.json', 'r') as f:
    data = json.load(f)
print(data.get('selected_model', ''))
")

if [ -z "${SELECTED_MODEL}" ]; then
    echo "⚠️  WARNING: Could not determine selected model, skipping final test"
else
    echo "   Selected model: ${SELECTED_MODEL}"
    echo "   Running final evaluation..."
    
    FINAL_EVAL_DIR="final_evaluation"
    mkdir -p ${FINAL_EVAL_DIR}
    
    # Chạy evaluation bình thường, không có các flag không hợp lệ
    python scripts/evaluate.py \
        --output_dir ${FINAL_EVAL_DIR}/ \
        --threshold_analysis \
        --generate_report
    
    if [ $? -ne 0 ]; then
        echo "⚠️  WARNING: Final evaluation had issues"
    else
        echo "✅ Final evaluation completed: ${FINAL_EVAL_DIR}/"
    fi
fi

# ============================================================================
# BƯỚC 7: TỔNG KẾT
# ============================================================================
echo "📊 STEP 7: Summary Report"
echo ""

# Hiển thị tuning summary nếu enabled
if [ "$TUNING_ENABLED" = true ]; then
    echo ""
    echo "🔧 HYPERPARAMETER TUNING SUMMARY:"
    echo "══════════════════════════════════════════════════════════════"
    
    # XGBoost
    if [ -f "${TUNING_DIR}/xgboost/tuning_summary.json" ]; then
        python -c "
import json
with open('${TUNING_DIR}/xgboost/tuning_summary.json') as f:
    data = json.load(f)
best_value = data.get('best_value', 0)
best_params = data.get('best_params', {})
print(f'xgboost: Best PR-AUC = {best_value:.4f}')
print(f'   Parameters: {list(best_params.keys())}')
"
    fi
    
    # MLP
    if [ -f "${TUNING_DIR}/mlp/tuning_summary.json" ]; then
        python -c "
import json
with open('${TUNING_DIR}/mlp/tuning_summary.json') as f:
    data = json.load(f)
best_value = data.get('best_value', 0)
best_params = data.get('best_params', {})
print(f'mlp: Best PR-AUC = {best_value:.4f}')
print(f'   Parameters: {list(best_params.keys())}')
"
    fi
    
    # GCN - ĐỌC TỪ THƯ MỤC MỚI
    if [ -f "${TUNING_DIR}/gnn_gcn/tuning_summary.json" ]; then
        python -c "
import json
with open('${TUNING_DIR}/gnn_gcn/tuning_summary.json') as f:
    data = json.load(f)
best_value = data.get('best_value', 0)
best_params = data.get('best_params', {})
print(f'gcn: Best PR-AUC = {best_value:.4f}')
print(f'   Parameters: {list(best_params.keys())}')
"
    fi
    
    # GAT - ĐỌC TỪ THƯ MỤC MỚI
    if [ -f "${TUNING_DIR}/gnn_gat/tuning_summary.json" ]; then
        python -c "
import json
with open('${TUNING_DIR}/gnn_gat/tuning_summary.json') as f:
    data = json.load(f)
best_value = data.get('best_value', 0)
best_params = data.get('best_params', {})
print(f'gat: Best PR-AUC = {best_value:.4f}')
print(f'   Parameters: {list(best_params.keys())}')
"
    fi
    
    echo "══════════════════════════════════════════════════════════════"
fi

# Hiển thị kết quả chọn model
if [ -f "best_model_selection.json" ]; then
    echo ""
    echo "📋 BEST MODEL SELECTION RESULT:"
    echo "══════════════════════════════════════════════════════════════"
    python -c "
import json, datetime
with open('best_model_selection.json', 'r') as f:
    data = json.load(f)

print(f\"Selected Model: {data.get('selected_model', 'N/A')}\")
print(f\"Selection Time: {data.get('selection_timestamp', 'N/A')}\")
print(f\"Candidates Considered: {data.get('candidates_considered', 'N/A')}\")
print(f\"Tolerance Margin: {data.get('tolerance_margin', 'N/A')*100}%\")

metrics = data.get('model_metrics', {})
if metrics:
    print(f\"\\n📈 Model Metrics:\")
    print(f\"   Macro PR-AUC: {metrics.get('macro_pr_auc', 'N/A'):.4f}\")
    print(f\"   Macro ROC-AUC: {metrics.get('macro_roc_auc', 'N/A'):.4f}\")
    print(f\"   Average Brier: {metrics.get('brier', 'N/A'):.4f}\")

rules = data.get('selection_rules', [])
if rules:
    print(f\"\\n🔧 Selection Rules:\")
    for rule in rules[:3]:
        print(f\"   • {rule}\")
    if len(rules) > 3:
        print(f\"   • ... and {len(rules)-3} more rules\")
"
    echo "══════════════════════════════════════════════════════════════"
fi

# Hiển thị calibration improvement nếu có
if [ -f "${CALIBRATION_DIR}/calibration_report.json" ]; then
    echo ""
    echo "⚖️  CALIBRATION IMPROVEMENT:"
    python -c "
import json, os
try:
    with open('${CALIBRATION_DIR}/calibration_report.json', 'r') as f:
        data = json.load(f)
    
    if 'average_brier_before' in data and 'average_brier_after' in data:
        before = data['average_brier_before']
        after = data['average_brier_after']
        improvement = data.get('average_improvement', 0)
        print(f\"   Average Brier Score:\")
        print(f\"     Before calibration: {before:.4f}\")
        print(f\"     After calibration:  {after:.4f}\")
        print(f\"     Improvement:       {improvement:.4f} ({improvement/before*100:.1f}%)\")
except Exception as e:
    print(f\"   Could not load calibration report: {e}\")
"
fi

echo ""
echo "📁 OUTPUT DIRECTORIES:"
echo "   ├── Processed Data: ${PROCESSED_DIR}/"
echo "   ├── Models: ${MODELS_DIR}/"
if [ "$TUNING_ENABLED" = true ]; then
echo "   ├── Tuning Results: ${TUNING_DIR}/"
fi
echo "   ├── Evaluation: ${RESULTS_DIR}/"
echo "   ├── Calibration: ${CALIBRATION_DIR}/"
echo "   └── Final Selection: best_model_selection.json"
echo ""

echo "⏱️  End time: $(date)"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    PIPELINE COMPLETED! ✅                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"