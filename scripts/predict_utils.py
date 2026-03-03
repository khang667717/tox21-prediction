"""
Prediction utilities for Tox21 models.
Used for inference, evaluation, calibration, and deployment.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any


# ============================================================
# Load probabilities
# ============================================================

def load_test_probabilities(model_dir: Path) -> np.ndarray:
    """
    Load saved test probabilities.
    Expected file: test_probs.npy
    Shape: (n_samples, n_tasks)
    """
    prob_path = model_dir / "test_probs.npy"

    if not prob_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prob_path}")

    probs = np.load(prob_path)

    if probs.ndim != 2:
        raise ValueError(f"Invalid probs shape {probs.shape}, expected 2D array")

    return probs


# ============================================================
# Mask-aware prediction handling
# ============================================================

def apply_mask(y_prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero-out predictions where mask == 0
    """
    if y_prob.shape != mask.shape:
        raise ValueError("Mask shape does not match prediction shape")

    return y_prob * mask


# ============================================================
# Metadata helpers
# ============================================================

def load_tasks(data_dir: Path) -> list[str]:
    tasks_path = data_dir / "tasks.json"
    with open(tasks_path) as f:
        return json.load(f)


def load_model_info(model_dir: Path) -> Dict[str, Any]:
    """
    Load optional model metadata.
    """
    info_path = model_dir / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f)
    return {}