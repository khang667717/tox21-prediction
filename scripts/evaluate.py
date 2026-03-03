import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
import datetime


# ---------- Load metrics helpers ----------

def load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_baseline_metrics(model_dir: Path) -> Dict[str, Any] | None:
    return load_json(model_dir / "xgboost_metrics.json")


def load_nn_metrics(model_dir: Path) -> Dict[str, Any] | None:
    return load_json(model_dir / "test_results.json")


# ---------- Threshold optimization ----------

def threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, mask: np.ndarray, 
                      tasks: List[str], output_dir: Path) -> Dict[str, Dict[str, float]]:
    """Find optimal threshold for each task based on F1."""
    print("\n🎯 Threshold analysis...")
    optimal_thresholds = {}
    
    for i, task in enumerate(tasks):
        task_mask = mask[:, i].astype(bool)
        if task_mask.sum() == 0:
            continue
        yt = y_true[task_mask, i]
        yp = y_prob[task_mask, i]
        
        thresholds = np.linspace(0.1, 0.9, 17)
        f1_scores = []
        for th in thresholds:
            pred = (yp >= th).astype(int)
            f1_scores.append(f1_score(yt, pred))
        
        best_idx = np.argmax(f1_scores)
        optimal_thresholds[task] = {
            'threshold': thresholds[best_idx],
            'f1': f1_scores[best_idx]
        }
    
    # Plot (first 6 tasks for readability)
    plt.figure(figsize=(10, 6))
    for i, task in enumerate(tasks[:6]):
        task_mask = mask[:, i].astype(bool)
        if task_mask.sum() == 0:
            continue
        yt = y_true[task_mask, i]
        yp = y_prob[task_mask, i]
        
        thresholds = np.linspace(0.1, 0.9, 17)
        f1_scores = [f1_score(yt, (yp >= th).astype(int)) for th in thresholds]
        plt.plot(thresholds, f1_scores, label=task)
    
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Threshold (Validation)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / 'threshold_optimization.png', dpi=300)
    plt.close()
    
    with open(output_dir / 'optimal_thresholds.json', 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    
    print(f"✅ Threshold analysis complete. Saved to {output_dir}")
    return optimal_thresholds


# ---------- Report generation ----------

def generate_full_report(df: pd.DataFrame, calibration_data: Optional[Dict], 
                        bigdata_summary: Dict, threshold_opt: Optional[Dict],
                        output_dir: Path) -> None:
    """Write comprehensive MARKDOWN report.
    FIXED: Handle missing tabulate dependency.
    """
    report_lines = []
    report_lines.append("# 🧪 Tox21 Project – Final Evaluation Report\n")
    report_lines.append(f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    
    # 1. Model comparison
    report_lines.append("## 1. Model Performance Comparison\n")
    
    # FIX: Try to use to_markdown, fallback to CSV if tabulate not installed
    try:
        report_lines.append(df.to_markdown(index=False) + "\n")
    except ImportError:
        print("⚠️ tabulate not installed. Using CSV format in report.")
        report_lines.append("```\n")
        report_lines.append(df.to_csv(index=False))
        report_lines.append("```\n")
    
    # 2. Best model
    best = df.loc[df['Macro PR-AUC'].idxmax()]
    report_lines.append(f"**🏆 Best model:** {best['Model']} (Macro PR-AUC = {best['Macro PR-AUC']:.4f})\n")
    
    # 3. Calibration
    report_lines.append("## 2. Calibration\n")
    if calibration_data:
        report_lines.append("| Model | Brier Score |\n|-------|-------------|\n")
        for model, cal in calibration_data.items():
            brier = cal.get('brier', 'N/A')
            if brier is not None and brier != 'N/A':
                report_lines.append(f"| {model} | {brier:.4f} |\n")
            else:
                report_lines.append(f"| {model} | N/A |\n")
    else:
        report_lines.append("*No calibration data available.*\n")
    
    # 4. Threshold analysis
    report_lines.append("## 3. Optimal Thresholds (Validation F1)\n")
    if threshold_opt:
        report_lines.append("| Task | Optimal Threshold | F1 |\n|------|------------------|-----|\n")
        for task, vals in list(threshold_opt.items())[:10]:  # top 10
            report_lines.append(f"| {task} | {vals['threshold']:.2f} | {vals['f1']:.3f} |\n")
        report_lines.append("\n![Threshold optimization](threshold_optimization.png)\n")
    else:
        report_lines.append("*No threshold analysis performed.*\n")
    
    # 5. Big Data demonstration
    report_lines.append("## 4. Big Data Integration\n")
    if bigdata_summary:
        for key, val in bigdata_summary.items():
            report_lines.append(f"- **{key}:** {val}\n")
    else:
        report_lines.append("*Spark/Dask was not used in this run.*\n")
    
    # 6. Conclusion
    report_lines.append("## 5. Conclusion & Limitations\n")
    report_lines.append("""
- **Strengths:** Multiple models (XGBoost, MLP, GCN, GAT) with thorough hyperparameter tuning; calibration analysis; interpretability (SHAP, gradients); FastAPI deployment.
- **Limitations:** No data augmentation; ensemble not yet implemented; Big Data demonstration is minimal but shows distributed featurization.
- **Future work:** Add ensemble, test on larger dataset with full Spark cluster.
""")
    
    # Save
    report_path = output_dir / 'REPORT.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"✅ Report saved: {report_path}")

# ---------- Comparison ----------

def compare_models(
    baseline_dir: Path,
    mlp_dir: Path,
    gnn_dirs: List[Path]
) -> pd.DataFrame:

    rows = []

    # Baseline
    baseline = load_baseline_metrics(baseline_dir)
    if baseline:
        rows.append({
            "Model": "XGBoost (ECFP4)",
            "Type": "Baseline",
            "Macro PR-AUC": baseline["test"]["macro_pr_auc"],
            "Macro ROC-AUC": baseline["test"]["macro_roc_auc"],
            "Micro PR-AUC": baseline["test"]["micro_pr_auc"],
            "Micro ROC-AUC": baseline["test"]["micro_roc_auc"],
        })

    # Descriptor MLP
    mlp = load_nn_metrics(mlp_dir)
    if mlp:
        rows.append({
            "Model": "Descriptor MLP",
            "Type": "Neural Network",
            "Macro PR-AUC": mlp["test_metrics"]["macro_pr_auc"],
            "Macro ROC-AUC": mlp["test_metrics"]["macro_roc_auc"],
            "Micro PR-AUC": mlp["test_metrics"]["micro_pr_auc"],
            "Micro ROC-AUC": mlp["test_metrics"]["micro_roc_auc"],
        })

    # GNNs
    for gnn_dir in gnn_dirs:
        gnn = load_nn_metrics(gnn_dir)
        if not gnn:
            continue

        rows.append({
            "Model": f"GNN ({gnn_dir.name.upper()})",
            "Type": "GNN",
            "Macro PR-AUC": gnn["test_metrics"]["macro_pr_auc"],
            "Macro ROC-AUC": gnn["test_metrics"]["macro_roc_auc"],
            "Micro PR-AUC": gnn["test_metrics"]["micro_pr_auc"],
            "Micro ROC-AUC": gnn["test_metrics"]["micro_roc_auc"],
        })

    return pd.DataFrame(rows)


# ---------- Plot ----------

def plot_comparison(df: pd.DataFrame, save_path: Path):
    metrics = ["Macro PR-AUC", "Macro ROC-AUC", "Micro PR-AUC", "Micro ROC-AUC"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        ax.barh(df["Model"], df[metric])
        ax.set_title(metric)
        ax.set_xlim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved plot → {save_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", default="models/baseline")
    parser.add_argument("--mlp_dir", default="models/descriptor_mlp")
    parser.add_argument("--gnn_dirs", nargs="+", default=["models/gnn/gcn", "models/gnn/gat"])
    parser.add_argument("--output_dir", default="reports/models")
    parser.add_argument("--threshold_analysis", action="store_true",
                        help="Run threshold optimization on validation set")
    parser.add_argument("--generate_report", action="store_true",
                        help="Generate comprehensive MARKDOWN report")
    parser.add_argument("--tasks_file", default="data/processed/tasks.json",
                        help="Path to tasks.json file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model comparison data
    df = compare_models(
        Path(args.baseline_dir),
        Path(args.mlp_dir),
        [Path(p) for p in args.gnn_dirs],
    )

    if df.empty:
        raise RuntimeError("No model metrics were loaded.")

    # Save basic comparison outputs
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    df.to_json(output_dir / "model_comparison.json", orient="records", indent=2)
    plot_comparison(df, output_dir / "model_comparison.png")

    best = df.loc[df["Macro PR-AUC"].idxmax()]
    with open(output_dir / "best_model.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    print(df)
    print(f"\nBest model: {best['Model']} (Macro PR-AUC={best['Macro PR-AUC']:.4f})")

    # ---- Threshold analysis (if requested) ----
    threshold_opt = None
    if args.threshold_analysis:
        # Load tasks
        tasks = []
        tasks_path = Path(args.tasks_file)
        if tasks_path.exists():
            with open(tasks_path) as f:
                tasks = json.load(f)
            print(f"✅ Loaded {len(tasks)} tasks from {tasks_path}")
        else:
            print(f"⚠️ Tasks file not found: {tasks_path}")
        
        # Load validation predictions from best model (Descriptor MLP by default)
        val_probs_path = Path(args.mlp_dir) / 'val_probs.npy'
        if val_probs_path.exists() and tasks:
            val_probs = np.load(val_probs_path)
            data_dir = Path('data/processed')
            y_val_path = data_dir / 'val.csv'
            mask_val_path = data_dir / 'val_mask.npy'
            
            if y_val_path.exists() and mask_val_path.exists():
                y_val = pd.read_csv(y_val_path)[tasks].values
                y_val = np.nan_to_num(y_val)
                mask_val = np.load(mask_val_path)
                
                threshold_opt = threshold_analysis(y_val, val_probs, mask_val, tasks, output_dir)
            else:
                print("⚠️ Validation data files not found, skipping threshold analysis")
        else:
            print("⚠️ Validation probabilities not found, skipping threshold analysis")

    # ---- Big Data summary (read from saved json) ----
    bigdata_summary = {}
    spark_result_path = Path('models/baseline/spark_baseline_results.json')
    if spark_result_path.exists():
        with open(spark_result_path) as f:
            spark_res = json.load(f)
        bigdata_summary['Spark MLlib LogisticRegression PR-AUC'] = str(spark_res)
        bigdata_summary['Featurization'] = 'Used Spark UDF (distributed)'
    else:
        bigdata_summary['Spark'] = 'Not executed'

    # ---- Generate report ----
    if args.generate_report:
        calibration_data = None
        cal_path = Path('calibration_results/calibration_metrics.json')
        if cal_path.exists():
            with open(cal_path) as f:
                calibration_data = json.load(f)
            print("✅ Loaded calibration data")
        else:
            print("⚠️ Calibration data not found")
        
        generate_full_report(df, calibration_data, bigdata_summary,
                           threshold_opt, output_dir)


if __name__ == "__main__":
    main()