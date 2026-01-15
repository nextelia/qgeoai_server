"""
Confusion matrix computation for semantic segmentation

Provides pixel-wise confusion matrix calculation for multi-class segmentation.
"""

import numpy as np


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix from predictions and targets.
    
    Pixel-wise confusion matrix for semantic segmentation.
    Rows = true class, Columns = predicted class.
    
    Args:
        predictions: Flattened predicted class indices (N,)
        targets: Flattened ground truth class indices (N,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes x num_classes)
        cm[i, j] = number of pixels with true class i predicted as class j
    """
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Populate confusion matrix
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            cm[true_class, pred_class] = np.sum(
                (targets == true_class) & (predictions == pred_class)
            )
    
    return cm