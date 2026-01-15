"""
Qualitative examples generation for training reports

Provides functions to:
- Generate inference examples from validation set
- Colorize segmentation masks
- Denormalize images for visualization
- Save triplet examples (Input / GT / Prediction)
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from core.training.config import(
    QUALITATIVE_HIGH_IOU_THRESHOLD,
    QUALITATIVE_LOW_IOU_THRESHOLD,
    QUALITATIVE_MEDIUM_IOU_MARGIN,
    QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY,
    QUALITATIVE_MAX_IMAGES_FOR_INFERENCE,
    QUALITATIVE_EXAMPLE_TILE_SIZE
)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize image tensor (ImageNet normalization).
    
    Args:
        image_tensor: Image tensor (C, H, W) normalized with ImageNet stats
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image as uint8 numpy array (H, W, 3)
    """
    # Convert to numpy and reorder to (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image_np = image_np * std + mean
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    
    return image_np


def colorize_mask(
    mask: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Convert class mask to RGB image using class colors.
    
    Args:
        mask: Class mask (H, W) with integer class IDs
        class_colors: Dict mapping class_id -> (R, G, B)
        
    Returns:
        RGB image (H, W, 3)
    """
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        colored[mask == class_id] = color
    
    return colored


def calculate_image_score(
    mask_gt: np.ndarray,
    pred_mask: np.ndarray,
    num_classes: int,
    task: str = 'semantic_segmentation'
) -> float:
    """
    Calculate image quality score.
    
    For semantic segmentation: pixel-wise IoU
    For instance segmentation: pixel-wise IoU (proxy for detection quality)
    
    Args:
        mask_gt: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        num_classes: Number of classes
        task: Task type
        
    Returns:
        Quality score (0-1)
    """
    # For both tasks, use pixel-wise IoU as a simple quality proxy
    ious = []
    
    for class_id in range(num_classes):
        gt_mask_class = (mask_gt == class_id)
        pred_mask_class = (pred_mask == class_id)
        
        intersection = np.logical_and(gt_mask_class, pred_mask_class).sum()
        union = np.logical_or(gt_mask_class, pred_mask_class).sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def save_example_triplet(
    image: torch.Tensor,
    mask_gt: np.ndarray,
    pred_mask: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]],
    output_dir: Path,
    example_idx: int
) -> Dict[str, str]:
    """
    Save triplet images (input, GT, prediction) for one example.
    
    Args:
        image: Input image tensor (C, H, W)
        mask_gt: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        class_colors: Color mapping for masks
        output_dir: Directory to save images
        example_idx: Index for naming files
        
    Returns:
        Dict with relative paths to saved images
    """
    # Denormalize input image
    image_np = denormalize_image(image)
    
    # Convert masks to colored images
    gt_colored = colorize_mask(mask_gt, class_colors)
    pred_colored = colorize_mask(pred_mask, class_colors)
    
    # Resize to display size
    image_pil = Image.fromarray(image_np).resize(
        (QUALITATIVE_EXAMPLE_TILE_SIZE, QUALITATIVE_EXAMPLE_TILE_SIZE), 
        Image.LANCZOS
    )
    gt_pil = Image.fromarray(gt_colored).resize(
        (QUALITATIVE_EXAMPLE_TILE_SIZE, QUALITATIVE_EXAMPLE_TILE_SIZE),
        Image.NEAREST  # Nearest for masks to keep sharp edges
    )
    pred_pil = Image.fromarray(pred_colored).resize(
        (QUALITATIVE_EXAMPLE_TILE_SIZE, QUALITATIVE_EXAMPLE_TILE_SIZE),
        Image.NEAREST
    )
    
    # Save images
    input_path = output_dir / f'example_{example_idx}_input.png'
    gt_path = output_dir / f'example_{example_idx}_gt.png'
    pred_path = output_dir / f'example_{example_idx}_pred.png'
    
    image_pil.save(input_path)
    gt_pil.save(gt_path)
    pred_pil.save(pred_path)
    
    # Return relative paths (from report root)
    category_name = output_dir.name
    return {
        'input': f'report_assets/{category_name}/example_{example_idx}_input.png',
        'gt': f'report_assets/{category_name}/example_{example_idx}_gt.png',
        'pred': f'report_assets/{category_name}/example_{example_idx}_pred.png'
    }


def generate_qualitative_examples(
    model: torch.nn.Module,
    val_dataset: Any,
    device: str,
    class_names: List[str],
    class_colors: Dict[int, Tuple[int, int, int]],
    assets_dir: Path,
    task: str = 'semantic_segmentation'
) -> Dict[str, Any]:
    """
    Generate qualitative examples by running inference on validation set.
    
    Selects examples based on per-image IoU:
    - High IoU (≥0.90): Best predictions
    - Medium IoU: Around model's average IoU
    - Low IoU (≤0.60): Challenging cases
    
    Also computes confusion matrix from all processed images.
    
    Args:
        model: Trained model (in eval mode)
        val_dataset: Validation dataset
        device: Device (cuda/cpu)
        class_names: List of class names
        class_colors: Dict mapping class_id -> RGB tuple
        assets_dir: Directory to save example images
        
    Returns:
        Dict with keys 'high', 'medium', 'low' containing example metadata,
        and 'confusion_matrix' with the computed confusion matrix
    """
    model.eval()
    
    # Storage for examples
    examples_data = {
        'high': [],
        'medium': [],
        'low': []
    }
    
    # Storage for all processed images
    all_results = []
    all_preds = []
    all_targets = []
    
    # Process validation images
    num_to_process = min(len(val_dataset), QUALITATIVE_MAX_IMAGES_FOR_INFERENCE)
    
    with torch.no_grad():
        for idx in range(num_to_process):
            try:
                # Get image and target (format differs by task)
                image, target = val_dataset[idx]

                # Move to device
                image_tensor = image.unsqueeze(0).to(device)

                # Extract ground truth mask
                if task == 'instance_segmentation':
                    # Target is a dict with 'masks' key
                    # For visualization, combine all instance masks into single semantic mask
                    masks_tensor = target['masks']  # (N, H, W) where N = num instances
                    labels_tensor = target['labels']  # (N,)
                    
                    # Create semantic mask by overlaying instances (last instance wins)
                    h, w = masks_tensor.shape[1], masks_tensor.shape[2]
                    mask_gt = np.zeros((h, w), dtype=np.uint8)
                    
                    for instance_idx in range(len(masks_tensor)):
                        instance_mask = masks_tensor[instance_idx].numpy()
                        label = labels_tensor[instance_idx].item()
                        mask_gt[instance_mask > 0] = label
                else:
                    # Semantic segmentation: target is already a mask tensor
                    mask_gt = target.numpy()
                
                # Inference
                if task == 'instance_segmentation':
                    # Mask R-CNN returns list of dicts
                    predictions = model(image_tensor)
                    pred = predictions[0]
                    
                    # Convert instance predictions to semantic mask
                    pred_masks = pred['masks'].cpu()  # (N, 1, H, W)
                    pred_labels = pred['labels'].cpu()  # (N,)
                    pred_scores = pred['scores'].cpu()  # (N,)
                    
                    # Create semantic mask from instances (use only confident predictions)
                    h, w = pred_masks.shape[2], pred_masks.shape[3]
                    pred_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    for instance_idx in range(len(pred_masks)):
                        if pred_scores[instance_idx] > 0.5:  # Confidence threshold
                            instance_mask = pred_masks[instance_idx, 0].numpy()
                            label = pred_labels[instance_idx].item()
                            pred_mask[instance_mask > 0.5] = label
                else:
                    # Semantic segmentation: argmax on class dimension
                    output = model(image_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                
                # Calculate per-image IoU
                score = calculate_image_score(mask_gt, pred_mask, len(class_names), task)
                
                # Store result
                all_results.append({
                    'idx': idx,
                    'score': score,
                    'image': image,
                    'mask_gt': mask_gt,
                    'pred_mask': pred_mask
                })
                
                # Accumulate for confusion matrix
                all_preds.append(pred_mask.flatten())
                all_targets.append(mask_gt.flatten())
                
            except Exception as e:
                print(f"Warning: Failed to process image {idx}: {e}")
                continue
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Select examples for each category
    high_examples = [r for r in all_results if r['score'] >= QUALITATIVE_HIGH_IOU_THRESHOLD]
    low_examples = [r for r in all_results if r['score'] <= QUALITATIVE_LOW_IOU_THRESHOLD]
    
    # Medium examples: around median score
    if all_results:
        median_iou = all_results[len(all_results) // 2]['score']
        medium_examples = [
            r for r in all_results 
            if abs(r['score'] - median_iou) <= QUALITATIVE_MEDIUM_IOU_MARGIN
        ]
    else:
        medium_examples = []
    
    # Select top N examples from each category
    selected_high = high_examples[:QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY]
    selected_medium = medium_examples[:QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY]
    selected_low = low_examples[:QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY]
    
    # Fallback for empty categories
    if len(selected_high) == 0 and len(all_results) > 0:
        selected_high = all_results[:1]
        examples_data['high_is_fallback'] = True
    else:
        examples_data['high_is_fallback'] = False
    
    if len(selected_low) == 0 and len(all_results) > 0:
        selected_low = all_results[-1:]
        examples_data['low_is_fallback'] = True
    else:
        examples_data['low_is_fallback'] = False
    
    if len(selected_medium) < QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY and len(all_results) > 0:
        mid_start = max(0, len(all_results) // 2 - QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY // 2)
        mid_end = min(len(all_results), mid_start + QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY)
        selected_medium = all_results[mid_start:mid_end]
    
    # Save example triplets
    examples_data['high'] = _save_category_examples(
        selected_high, 'high', class_colors, assets_dir
    )
    examples_data['medium'] = _save_category_examples(
        selected_medium, 'medium', class_colors, assets_dir
    )
    examples_data['low'] = _save_category_examples(
        selected_low, 'low', class_colors, assets_dir
    )
    
    # Compute confusion matrix (only for semantic segmentation)
    if task == 'semantic_segmentation' and all_preds and all_targets:
        from core.training.metrics import compute_confusion_matrix
        
        confusion_matrix = compute_confusion_matrix(
            predictions=np.concatenate(all_preds),
            targets=np.concatenate(all_targets),
            num_classes=len(class_names)
        )
        examples_data['confusion_matrix'] = confusion_matrix
    else:
        examples_data['confusion_matrix'] = None

    return examples_data


def _save_category_examples(
    examples: List[Dict],
    category: str,
    class_colors: Dict[int, Tuple[int, int, int]],
    assets_dir: Path
) -> List[Dict[str, str]]:
    """
    Save all examples for a category.
    
    Args:
        examples: List of example data dicts
        category: Category name ('high', 'medium', 'low')
        class_colors: Color mapping for masks
        assets_dir: Base assets directory
        
    Returns:
        List of dicts with paths and metadata
    """
    # Create category directory
    category_dir = assets_dir / f'qualitative_{category}'
    category_dir.mkdir(exist_ok=True)
    
    saved_examples = []
    
    for i, example in enumerate(examples):
        paths = save_example_triplet(
            image=example['image'],
            mask_gt=example['mask_gt'],
            pred_mask=example['pred_mask'],
            class_colors=class_colors,
            output_dir=category_dir,
            example_idx=i
        )
        
        # Add score to metadata
        paths['score'] = f"{example['score']:.3f}"
        saved_examples.append(paths)
    
    return saved_examples