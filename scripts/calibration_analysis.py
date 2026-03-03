
#!/usr/bin/env python3
"""
Calibration analysis for trained Tox21 models.
Requires saved probability predictions (*.npy).
"""

import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


# ---------- Core ----------

def compute_calibration(y_true, y_prob, mask, n_bins=10):
    y_true = y_true[mask == 1]
    y_prob = y_prob[mask == 1]

    if y_true.size == 0:
        return None, None, None

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    return prob_true, prob_pred, brier


# ---------- Plot ----------

def plot_calibration(results: Dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))

    for name, r in results.items():
        if r["prob_true"] is None:
            continue
        ax.plot(
            r["prob_pred"],
            r["prob_true"],
            marker="o",
            label=f"{name} (Brier={r['brier']:.3f})"
        )

    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curves")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved calibration plot → {save_path}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--output_dir", default="calibration_results")  # FIX: Changed default
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels & mask
    test_df = pd.read_csv(data_dir / "test.csv")
    test_mask = np.load(data_dir / "test_mask.npy")

    with open(data_dir / "tasks.json") as f:
        tasks = json.load(f)

    y_true = np.nan_to_num(test_df[tasks].values)

    models = {
        "XGBoost": Path(args.models_dir) / "baseline" / "test_probs.npy",
        "MLP": Path(args.models_dir) / "descriptor_mlp" / "test_probs.npy",
        "GCN": Path(args.models_dir) / "gnn_gcn" / "test_probs.npy",   # ✅ ĐÚNG
        "GAT": Path(args.models_dir) / "gnn_gat" / "test_probs.npy",   # ✅ ĐÚNG
    }

    results = {}

    for name, prob_path in models.items():
        if not prob_path.exists():
            print(f"[SKIP] {name}: missing {prob_path}")
            results[name] = {"prob_true": None, "prob_pred": None, "brier": None}
            continue

        y_prob = np.load(prob_path)
        
        # FIX: Handle empty probability arrays
        if y_prob.shape[0] == 0:
            print(f"[SKIP] {name}: empty probability array")
            results[name] = {"prob_true": None, "prob_pred": None, "brier": None}
            continue
            
        assert y_prob.shape == y_true.shape, f"Shape mismatch for {name}: {y_prob.shape} vs {y_true.shape}"

        pt, pp, brier = compute_calibration(y_true, y_prob, test_mask)
        results[name] = {
            "prob_true": pt,
            "prob_pred": pp,
            "brier": brier,
        }

        print(f"{name}: Brier = {brier:.4f}" if brier is not None else f"{name}: No calibration data")

    plot_calibration(results, output_dir / "calibration.png")

    # Save metrics
    out = {}
    for k, v in results.items():
        out[k] = {
            "brier": None if v["brier"] is None else float(v["brier"]),
            "prob_true": None if v["prob_true"] is None else v["prob_true"].tolist(),
            "prob_pred": None if v["prob_pred"] is None else v["prob_pred"].tolist(),
        }

    with open(output_dir / "calibration_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print("[OK] Saved calibration_metrics.json")


if __name__ == "__main__":
    main()