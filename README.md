# 🧪 Tox21 Molecular Toxicity Prediction

A professional-grade machine learning pipeline for predicting molecular toxicity using the Tox21 dataset. This project leverages multiple architectures, from traditional XGBoost to deep learning and Graph Neural Networks (GNNs).

## 📊 Overview
The **Tox21 (Toxicology in the 21st Century)** challenge aims to identify chemical compounds that can disrupt biological pathways. This repository provides a complete, automated workflow for training, evaluating, and deploying models to predict 12 different toxicity tasks.

## 🚀 Key Features
- **Multi-Model Support**: XGBoost (Fingerprints), MLP (Descriptors), and GNN (Graph-based: GCN, GAT, GINE).
- **Automated Pipeline**: End-to-end automation from raw data to a deployed API.
- **Robust Methodology**: Scaffold splitting (Bemis-Murcko) and calibration analysis (Brier scores).
- **Interpretability**: SHAP and gradient-based attribution to visualize toxicophores.
- **Production Ready**: FastAPI server for real-time predictions.

## 📁 Project Structure
```text
.
├── data/raw/              # Raw Tox21 dataset (tox21.csv)
├── data/processed/        # Processed splits and masks
├── scripts/               # Core execution scripts
│   ├── data_preprocess.py # Scaffold splitting and cleaning
│   ├── feature_extraction.py # RDKit feature engineering
│   ├── baseline_model.py  # XGBoost implementation
│   ├── descriptor_mlp.py  # Neural Network on descriptors
│   ├── gnn_model.py      # Graph Neural Networks (PyG)
│   └── evaluate.py        # Comparative metrics and reports
├── models/                # Saved model checkpoints and joblib files
├── results/               # Evaluation plots and comparison CSVs
├── FastAPI_fixed.py       # API deployment server
├── run_full_pipeline.sh   # Master shell script to run everything
└── requirements.txt       # Environment dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd tox21_project
   ```

2. **Setup environment**:
   ```bash
   python -m venv tox21_env
   source tox21_env/bin/activate
   pip install -r requirements.txt
   ```

## 🏃 Usage

### 1. Run the Full Pipeline
To run the entire workflow (preprocessing $\to$ training $\to$ evaluation $\to$ selection):
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh --tuning --tuning_trials=30
./run_full_pipeline.sh --tuning --tuning_trials=20  --spark (dùng spark và tunning)
./run_full_pipeline.sh --spark (khi không dùng tunning)
```
next run Post Pipeline
 ./run_post_pipeline.sh --all

### 2. Deploy the Prediction API
Once a model is selected, start the FastAPI server:
```bash
python FastAPI_fixed.py
```
Access the interactive documentation at `http://localhost:8000/docs`.

After that, run demo_predict_v2.py
```bash
python demo_predict_v2.py
```
(optional) use option 6 with threshold = 0.45 
### 3. Individual Components
- **Preprocessing**: `python scripts/data_preprocess.py --input data/raw/tox21.csv`
- **GNN Training**: `python scripts/gnn_model.py --gnn_type gcn --use_edge_features`
- **Interpretability**: `python interpretability.py --model_type gnn --model_dir models/gnn_gcn`

## 🧠 Methodology

### Data Splitting
We use **Scaffold Splitting** to ensure the training and testing sets contain different molecular backbones. This is more rigorous than random splitting and better reflects real-world generalizability in drug discovery.

### Feature Engineering
- **Fingerprints**: Morgan (ECFP4) bit/count vectors.
- **Descriptors**: 180+ RDKit physical and chemical properties.
- **Graphs**: Node features (Atomic number, charge, rings) + Edge features (Bond type).

### Evaluation
Models are evaluated on:
- **Macro PR-AUC**: Primary metric due to high class imbalance.
- **ROC-AUC**: For overall discriminative power.
- **Brier Score**: To measure calibration quality of probabilities.

## ⚖️ License
This project is for educational and research purposes. Data is provided by the NIH NCATS Tox21 Challenge.


