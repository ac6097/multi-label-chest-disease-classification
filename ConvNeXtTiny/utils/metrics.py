import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def multilabel_metrics(targets, probabilities, known, threshold=0.5):
    """Compute metrics from already Study-aggregated predictions and labels."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    known = np.asarray(known, dtype=bool)
    if targets.shape != probabilities.shape or known.shape != targets.shape:
        raise ValueError("targets, probabilities, and known must have the same [Study, class] shape")
    per_label, aucs, f1s = [], [], []
    for index in range(targets.shape[1]):
        mask = known[:, index]
        y = targets[mask, index].astype(int)
        p = probabilities[mask, index]
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        auc = roc_auc_score(y, p) if n_pos and n_neg else np.nan
        auprc = average_precision_score(y, p) if y.size and n_pos else np.nan
        predicted = p >= threshold
        tp = int(np.sum((y == 1) & predicted))
        fp = int(np.sum((y == 0) & predicted))
        tn = int(np.sum((y == 0) & ~predicted))
        fn = int(np.sum((y == 1) & ~predicted))
        precision = tp / (tp + fp) if y.size and tp + fp else (0.0 if y.size else np.nan)
        recall = tp / n_pos if n_pos else (0.0 if y.size else np.nan)
        f1 = (2.0 * precision * recall / (precision + recall)
              if y.size and precision + recall else (0.0 if y.size else np.nan))
        if np.isfinite(auc):
            aucs.append(float(auc))
        if np.isfinite(f1):
            f1s.append(float(f1))
        per_label.append({
            "auroc": float(auc), "auprc": float(auprc),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "n_valid": int(y.size), "n_positive": n_pos, "n_negative": n_neg,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else np.nan,
        "macro_f1": float(np.mean(f1s)) if f1s else np.nan,
        "per_label": per_label,
    }
