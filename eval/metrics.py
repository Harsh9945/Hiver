# eval/metrics.py
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

def compute_classification_metrics(y_true: List[str], y_pred: List[str], classes: List[str] = None) -> Dict[str, Any]:
    """
    Computes Accuracy, Macro-F1, Weighted-F1, per-class metrics, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class metrics
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))
        
    p_per, r_per, f1_per, supp_per = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = {}
    for idx, cls_name in enumerate(classes):
        per_class[cls_name] = {
            "precision": round(float(p_per[idx]), 4),
            "recall": round(float(r_per[idx]), 4),
            "f1": round(float(f1_per[idx]), 4),
            "support": int(supp_per[idx])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "classes": classes,
        "confusion_matrix": cm
    }

def compute_routing_metrics(
    y_true_decisions: List[str], 
    y_pred_decisions: List[str], 
    cost_fn: float = 5.0, 
    cost_fp: float = 1.0
) -> Dict[str, Any]:
    """
    Computes Routing Precision, Recall, F1 against 'escalate', and Cost-Weighted Routing Penalty.
    False Negative (FN) = Should escalate, but auto-handled (severe under-escalation).
    False Positive (FP) = Should auto-handle, but escalated (unnecessary queue congestion).
    """
    total = len(y_true_decisions)
    tp = sum(1 for t, p in zip(y_true_decisions, y_pred_decisions) if t == "escalate" and p == "escalate")
    fp = sum(1 for t, p in zip(y_true_decisions, y_pred_decisions) if t == "auto_handle" and p == "escalate")
    tn = sum(1 for t, p in zip(y_true_decisions, y_pred_decisions) if t == "auto_handle" and p == "auto_handle")
    fn = sum(1 for t, p in zip(y_true_decisions, y_pred_decisions) if t == "escalate" and p == "auto_handle")
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    expected_cost = (cost_fn * fn + cost_fp * fp) / float(total) if total > 0 else 0.0
    
    # Compare with 'Always Escalate' baseline
    auto_count = sum(1 for t in y_true_decisions if t == "auto_handle")
    always_escalate_cost = (cost_fp * auto_count) / float(total) if total > 0 else 0.0
    
    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "expected_unit_cost": round(float(expected_cost), 4),
        "always_escalate_cost": round(float(always_escalate_cost), 4)
    }

def compute_reply_metrics(groundedness_scores: List[float], similarities: List[float]) -> Dict[str, Any]:
    """
    Computes summary distribution of groundedness and retrieval similarities.
    """
    return {
        "mean_groundedness": round(float(np.mean(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "median_groundedness": round(float(np.median(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "min_groundedness": round(float(np.min(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "max_groundedness": round(float(np.max(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "mean_similarity": round(float(np.mean(similarities)), 4) if similarities else 0.0
    }
