"""Evaluation metrics and utilities."""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def calculate_metrics(predictions, targets, class_names=None):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities)
        targets (torch.Tensor): Ground truth labels
        class_names (list): Optional class names for reporting
        
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Get predicted classes
    pred_classes = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, pred_classes, average=None, zero_division=0
    )
    
    # Macro F1-score
    macro_f1 = np.mean(f1)
    
    # Accuracy
    accuracy = np.mean(pred_classes == targets)
    
    # Confusion matrix
    cm = confusion_matrix(targets, pred_classes)
    
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "confusion_matrix": cm
    }
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print evaluation metrics.
    
    Args:
        metrics (dict): Metrics dictionary from calculate_metrics
        class_names (list): Optional class names
    """
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    
    for i in range(len(metrics['precision'])):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision'][i]:.4f}")
        print(f"  Recall: {metrics['recall'][i]:.4f}")
        print(f"  F1-Score: {metrics['f1'][i]:.4f}")
        print(f"  Support: {int(metrics['support'][i])}")
    
    print("=" * 60 + "\n")
