"""
Semantic segmentation metrics

Provides pixel-wise metrics for semantic segmentation tasks:
- IoU (Intersection over Union / Jaccard Index)
- F1 Score (Dice Coefficient)
- Precision and Recall
- Pixel Accuracy
"""

import numpy as np
import torch
from typing import Tuple, List, Dict


def compute_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> float:
    """
    Compute Intersection over Union (IoU) for semantic segmentation.
    
    Also known as Jaccard Index. Computed as:
    IoU = TP / (TP + FP + FN)
    
    Args:
        predictions: Model predictions (B, C, H, W) - logits
        targets: Ground truth masks (B, H, W) - class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Mean IoU across all classes
    """
    # Get predicted classes
    preds = torch.argmax(predictions, dim=1)  # (B, H, W)
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignored indices
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
    
    # Compute IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            # Class not present in batch
            continue
        
        iou = intersection / union
        ious.append(iou.item())
    
    # Mean IoU
    return np.mean(ious) if ious else 0.0


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> float:
    """
    Compute F1 score (Dice coefficient) for semantic segmentation.
    
    F1 = 2 * Precision * Recall / (Precision + Recall)
       = 2 * TP / (2 * TP + FP + FN)
    
    Args:
        predictions: Model predictions (B, C, H, W) - logits
        targets: Ground truth masks (B, H, W) - class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Mean F1 score across all classes
    """
    # Get predicted classes
    preds = torch.argmax(predictions, dim=1)  # (B, H, W)
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignored indices
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
    
    # Compute F1 for each class
    f1_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        
        if (tp + fp + fn) == 0:
            # Class not present in batch
            continue
        
        f1 = (2 * tp) / (2 * tp + fp + fn)
        f1_scores.append(f1.item())
    
    # Mean F1
    return np.mean(f1_scores) if f1_scores else 0.0


def compute_precision_recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = -100
) -> Tuple[float, float]:
    """
    Compute Precision and Recall for semantic segmentation.
    
    Precision = TP / (TP + FP)  - "Of all predicted positive, how many are correct?"
    Recall = TP / (TP + FN)     - "Of all actual positive, how many did we find?"
    
    Args:
        predictions: Model predictions (B, C, H, W) - logits
        targets: Ground truth masks (B, H, W) - class indices
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        
    Returns:
        Tuple of (mean_precision, mean_recall) across all classes
    """
    # Get predicted classes
    preds = torch.argmax(predictions, dim=1)  # (B, H, W)
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignored indices
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
    
    # Compute Precision and Recall for each class
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        tp = (pred_cls & target_cls).sum().float()
        fp = (pred_cls & ~target_cls).sum().float()
        fn = (~pred_cls & target_cls).sum().float()
        
        # Precision
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            precisions.append(precision.item())
        
        # Recall
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            recalls.append(recall.item())
    
    # Mean Precision and Recall
    mean_precision = np.mean(precisions) if precisions else 0.0
    mean_recall = np.mean(recalls) if recalls else 0.0
    
    return mean_precision, mean_recall


def compute_pixel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        predictions: Model predictions (B, C, H, W) - logits
        targets: Ground truth masks (B, H, W) - class indices
        ignore_index: Index to ignore in computation
        
    Returns:
        Pixel accuracy as a float
    """
    # Get predicted classes
    preds = torch.argmax(predictions, dim=1)  # (B, H, W)
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignored indices
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
    
    # Compute accuracy
    correct = (preds == targets).sum().float()
    total = targets.numel()
    
    return (correct / total).item() if total > 0 else 0.0

# =============================================================================
# INSTANCE SEGMENTATION METRICS (MASK R-CNN)
# =============================================================================

def compute_instance_ap(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision (AP) for instance segmentation.
    
    AP@0.5 - Standard COCO metric for instance segmentation.
    Measures detection quality: how many predicted instances match ground truth?
    
    Args:
        predictions: List of prediction dicts from Mask R-CNN, each containing:
            - 'boxes': FloatTensor (N, 4) in [x1, y1, x2, y2] format
            - 'labels': Int64Tensor (N,) with class indices
            - 'scores': FloatTensor (N,) with confidence scores
            - 'masks': ByteTensor (N, H, W) with binary masks
        targets: List of target dicts, each containing:
            - 'boxes': FloatTensor (M, 4)
            - 'labels': Int64Tensor (M,)
            - 'masks': ByteTensor (M, H, W)
        num_classes: Number of classes (including background)
        iou_threshold: IoU threshold for matching (default: 0.5)
        
    Returns:
        Mean Average Precision across all classes
    """
    try:
        from torchvision.ops import box_iou
    except ImportError:
        raise ImportError("torchvision is required for instance segmentation metrics")
    
    # Collect all predictions and targets per class
    class_aps = []
    
    # Iterate over classes (excluding background class 0)
    for class_id in range(1, num_classes):
        all_scores = []
        all_tp = []
        num_gt = 0
        
        # Iterate over images
        for pred, target in zip(predictions, targets):
            # Filter predictions for this class
            pred_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            pred_masks = pred['masks'][pred_mask]
            
            # Filter targets for this class
            target_mask = target['labels'] == class_id
            target_boxes = target['boxes'][target_mask]
            target_masks = target['masks'][target_mask]
            
            num_gt += target_boxes.shape[0]
            
            if pred_boxes.shape[0] == 0:
                continue
            
            if target_boxes.shape[0] == 0:
                # False positives
                all_scores.extend(pred_scores.cpu().tolist())
                all_tp.extend([0] * pred_boxes.shape[0])
                continue
            
            # Compute IoU between predicted and target boxes
            ious = box_iou(pred_boxes, target_boxes)
            
            # For each prediction, find best matching target
            max_ious, matched_targets = ious.max(dim=1)
            
            # Mark as true positive if IoU >= threshold
            tp = (max_ious >= iou_threshold).cpu()
            
            # Handle multiple predictions matching same target (keep highest score)
            matched_targets_list = matched_targets.cpu().tolist()
            used_targets = set()
            
            for i, (score, is_tp, target_idx) in enumerate(zip(pred_scores, tp, matched_targets_list)):
                if is_tp and target_idx not in used_targets:
                    all_tp.append(1)
                    used_targets.add(target_idx)
                else:
                    all_tp.append(0)
                all_scores.append(score.item())
        
        if num_gt == 0:
            # No ground truth for this class
            continue
        
        if len(all_scores) == 0:
            # No predictions for this class
            class_aps.append(0.0)
            continue
        
        # Sort by confidence score (descending)
        sorted_indices = np.argsort(all_scores)[::-1]
        all_tp = np.array(all_tp)[sorted_indices]
        
        # Compute cumulative TP and FP
        cum_tp = np.cumsum(all_tp)
        cum_fp = np.cumsum(1 - all_tp)
        
        # Compute precision and recall
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / num_gt
        
        # Compute AP using 11-point interpolation (COCO style)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        class_aps.append(ap)
    
    # Mean AP across all classes
    return np.mean(class_aps) if class_aps else 0.0


def compute_instance_ar(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
    max_detections: int = 100
) -> float:
    """
    Compute Average Recall (AR) for instance segmentation.
    
    AR@100 - Maximum recall given 100 detections per image.
    Measures how many ground truth instances are detected.
    
    Args:
        predictions: List of prediction dicts from Mask R-CNN
        targets: List of target dicts
        num_classes: Number of classes (including background)
        iou_threshold: IoU threshold for matching (default: 0.5)
        max_detections: Maximum number of detections per image (default: 100)
        
    Returns:
        Mean Average Recall across all classes
    """
    try:
        from torchvision.ops import box_iou
    except ImportError:
        raise ImportError("torchvision is required for instance segmentation metrics")
    
    class_recalls = []
    
    # Iterate over classes (excluding background class 0)
    for class_id in range(1, num_classes):
        total_gt = 0
        total_matched = 0
        
        # Iterate over images
        for pred, target in zip(predictions, targets):
            # Filter predictions for this class (top max_detections by score)
            pred_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            
            # Keep only top-k predictions
            if pred_boxes.shape[0] > max_detections:
                top_k = torch.topk(pred_scores, k=max_detections)
                pred_boxes = pred_boxes[top_k.indices]
                pred_scores = pred_scores[top_k.indices]
            
            # Filter targets for this class
            target_mask = target['labels'] == class_id
            target_boxes = target['boxes'][target_mask]
            
            total_gt += target_boxes.shape[0]
            
            if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
                continue
            
            # Compute IoU between predicted and target boxes
            ious = box_iou(pred_boxes, target_boxes)
            
            # For each target, check if any prediction matches
            max_ious = ious.max(dim=0).values
            matched = (max_ious >= iou_threshold).sum().item()
            total_matched += matched
        
        if total_gt == 0:
            # No ground truth for this class
            continue
        
        recall = total_matched / total_gt
        class_recalls.append(recall)
    
    # Mean recall across all classes
    return np.mean(class_recalls) if class_recalls else 0.0


def compute_instance_precision_recall_f1(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F1 score for instance segmentation.
    
    These are computed at a fixed confidence threshold (default: 0.5).
    
    Args:
        predictions: List of prediction dicts from Mask R-CNN
        targets: List of target dicts
        num_classes: Number of classes (including background)
        iou_threshold: IoU threshold for matching (default: 0.5)
        score_threshold: Confidence score threshold (default: 0.5)
        
    Returns:
        Tuple of (mean_precision, mean_recall, mean_f1) across all classes
    """
    try:
        from torchvision.ops import box_iou
    except ImportError:
        raise ImportError("torchvision is required for instance segmentation metrics")
    
    class_precisions = []
    class_recalls = []
    class_f1s = []
    
    # Iterate over classes (excluding background class 0)
    for class_id in range(1, num_classes):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Iterate over images
        for pred, target in zip(predictions, targets):
            # Filter predictions for this class (above score threshold)
            pred_mask = (pred['labels'] == class_id) & (pred['scores'] >= score_threshold)
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            
            # Filter targets for this class
            target_mask = target['labels'] == class_id
            target_boxes = target['boxes'][target_mask]
            
            num_pred = pred_boxes.shape[0]
            num_target = target_boxes.shape[0]
            
            if num_pred == 0 and num_target == 0:
                continue
            
            if num_pred == 0:
                # All targets are false negatives
                total_fn += num_target
                continue
            
            if num_target == 0:
                # All predictions are false positives
                total_fp += num_pred
                continue
            
            # Compute IoU between predicted and target boxes
            ious = box_iou(pred_boxes, target_boxes)
            
            # Match predictions to targets (greedy matching)
            matched_targets = set()
            for i in range(num_pred):
                max_iou, matched_target = ious[i].max(dim=0)
                matched_target = matched_target.item()
                
                if max_iou >= iou_threshold and matched_target not in matched_targets:
                    total_tp += 1
                    matched_targets.add(matched_target)
                else:
                    total_fp += 1
            
            # Unmatched targets are false negatives
            total_fn += num_target - len(matched_targets)
        
        # Compute precision, recall, F1 for this class
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
            class_precisions.append(precision)
        
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
            class_recalls.append(recall)
        
        if total_tp + total_fp > 0 and total_tp + total_fn > 0:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_f1s.append(f1)
    
    # Mean across all classes
    mean_precision = np.mean(class_precisions) if class_precisions else 0.0
    mean_recall = np.mean(class_recalls) if class_recalls else 0.0
    mean_f1 = np.mean(class_f1s) if class_f1s else 0.0
    
    return mean_precision, mean_recall, mean_f1


def compute_mask_iou(
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU between predicted and target masks (for instance segmentation).
    
    Args:
        pred_masks: Predicted masks (N, H, W) - binary
        target_masks: Target masks (M, H, W) - binary
        
    Returns:
        IoU matrix (N, M)
    """
    # Flatten masks
    pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
    target_flat = target_masks.view(target_masks.shape[0], -1).float()
    
    # Compute intersection and union
    intersection = torch.mm(pred_flat, target_flat.t())
    
    pred_area = pred_flat.sum(dim=1, keepdim=True)
    target_area = target_flat.sum(dim=1, keepdim=True)
    union = pred_area + target_area.t() - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou