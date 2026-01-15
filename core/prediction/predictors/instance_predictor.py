"""
Instance Segmentation Predictor for QPredict

Handles inference for instance segmentation models (Mask R-CNN)
with object detection + segmentation masks and NMS for overlap handling.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional

from core.prediction.predictors.base_predictor import BasePredictor


class InstanceSegmentationPredictor(BasePredictor):
    """
    Predictor for instance segmentation models (Mask R-CNN) - server version.
    
    Handles:
    - Batch inference with instance predictions
    - Confidence filtering
    - NMS (Non-Maximum Suppression) between tiles
    
    Returns instance data as list of dicts (no file creation).
    
    Args:
        model: PyTorch model (Mask R-CNN)
        batch_size: Batch size for inference (recommend 1-2 for Mask R-CNN)
        device: Device (CPU/CUDA)
        model_info: Model configuration
        output_shape: (height, width) of the output
        confidence_threshold: Minimum confidence for detections (default: 0.5)
        nms_threshold: IoU threshold for NMS (default: 0.5)
        overlap: Overlap size (needed for margin calculation)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize instance segmentation predictor."""
        super().__init__(*args, **kwargs)
        
        # Extract instance-specific arguments
        self.confidence_threshold = self.kwargs.get('confidence_threshold', 0.5)
        self.nms_threshold = self.kwargs.get('nms_threshold', 0.5)
        self.overlap = self.kwargs.get('overlap', 0)  # For margin calculation
    
    def _initialize_accumulation(self):
        """
        Initialize structures for accumulating instance predictions.
        
        For instance segmentation:
        - accumulated_instances: List of all detected instances across tiles
          Each instance: {
              'bbox': [x_min, y_min, x_max, y_max],  # Global coordinates
              'mask_local': np.ndarray,  # Binary mask (H, W) in local coords
              'mask_offset': (y, x),  # Position in global coordinates
              'class_id': int,
              'score': float
          }
        """
        self.accumulated_instances = []
    
    def _process_batch(self, batch_tiles: list, batch_metadata: list):
        """
        Process a batch of tiles through Mask R-CNN.
        
        Mask R-CNN expects a list of images (not batched tensor).
        
        Args:
            batch_tiles: List of tile tensors
            batch_metadata: List of tile metadata dicts
        """
        # Mask R-CNN takes list of images
        images = [tile.to(self.device) for tile in batch_tiles]
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(images)
        
        # predictions is a list of dicts, one per image
        # Each dict: {'boxes': tensor, 'labels': tensor, 'scores': tensor, 'masks': tensor}
        
        # Process each prediction
        for pred, metadata in zip(predictions, batch_metadata):
            self._accumulate_prediction(pred, metadata)
    
    def _accumulate_prediction(self, prediction: Dict[str, torch.Tensor], metadata: Dict[str, Any]):
        """
        Accumulate instance predictions from a single tile.
        
        Converts tile-local coordinates to global raster coordinates.
        Filters by confidence threshold and centroid position.
        
        Args:
            prediction: Mask R-CNN output dict with keys:
                - 'boxes': (N, 4) bounding boxes in tile coordinates
                - 'labels': (N,) class labels (1-indexed)
                - 'scores': (N,) confidence scores
                - 'masks': (N, 1, H, W) binary masks (probabilities > 0.5)
            metadata: Tile metadata with position info
                Required keys:
                - tile_width, tile_height: actual tile dimensions
                - pad_left, pad_top: padding applied
                - out_x, out_y: position in output array
        """
        # Get tile info
        tile_width = metadata['tile_width']
        tile_height = metadata['tile_height']
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']
        out_x = metadata['out_x']
        out_y = metadata['out_y']
        
        # Move predictions to CPU
        boxes = prediction['boxes'].cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        labels = prediction['labels'].cpu().numpy()  # (N,) class IDs (1-indexed)
        scores = prediction['scores'].cpu().numpy()  # (N,) confidence scores
        masks = prediction['masks'].cpu().numpy()  # (N, 1, H_padded, W_padded)
        
        # Filter by confidence threshold
        keep = scores >= self.confidence_threshold
        
        # No detections above threshold
        if not np.any(keep):
            return  
        
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks = masks[keep]
        
        # Process each instance
        for i in range(len(boxes)):
            box = boxes[i]  # [x1, y1, x2, y2] in padded tile coordinates
            label = labels[i]
            score = scores[i]
            mask = masks[i, 0]  # (H_padded, W_padded) probabilities
            
            # Crop mask to remove padding
            mask_cropped = mask[
                pad_top:pad_top + tile_height,
                pad_left:pad_left + tile_width
            ]
            
            # Threshold mask to binary (Mask R-CNN outputs probabilities)
            mask_binary = (mask_cropped > 0.5).astype(np.uint8)
            
            # Skip if mask is empty after thresholding
            if not np.any(mask_binary):
                continue

            # CRITICAL: Filter by centroid position (keep only centered detections)
            # This prevents duplicate detections at tile edges
            local_cx = (box[0] + box[2]) / 2
            local_cy = (box[1] + box[3]) / 2
            
            # Define margin (detections in this border zone are discarded)
            margin = max(32, self.overlap // 4)
            
            # Keep only if centroid is in central zone
            if not (margin <= local_cx - pad_left < tile_width - margin and
                    margin <= local_cy - pad_top < tile_height - margin):
                continue  # Skip this detection (edge detection, likely duplicate)
            
            # Adjust bounding box coordinates to global
            box_adjusted = np.array([
                box[0] - pad_left + out_x,  # x_min
                box[1] - pad_top + out_y,   # y_min
                box[2] - pad_left + out_x,  # x_max
                box[3] - pad_top + out_y    # y_max
            ])
            
            # Clip to output extent
            box_clipped = np.array([
                max(0, box_adjusted[0]),
                max(0, box_adjusted[1]),
                min(self.output_shape[1], box_adjusted[2]),
                min(self.output_shape[0], box_adjusted[3])
            ])
            
            # Skip if box is invalid after clipping
            if box_clipped[2] <= box_clipped[0] or box_clipped[3] <= box_clipped[1]:
                continue
            
            # Store instance with local mask + offset (memory efficient)
            instance = {
                'bbox': box_clipped,
                'mask_local': mask_binary,  # Small local mask
                'mask_offset': (out_y, out_x),  # Position in global coordinates
                'class_id': int(label),
                'score': float(score)
            }
            
            self.accumulated_instances.append(instance)
    
    def _finalize_predictions(self) -> List[Dict[str, Any]]:
        """
        Finalize predictions by applying NMS across tiles.
        
        Mask R-CNN may detect the same object in multiple overlapping tiles.
        We use NMS (Non-Maximum Suppression) to remove duplicates.
        
        Returns:
            List of final instances after NMS
        """
        if not self.accumulated_instances:
            return []
        
        # Apply NMS per class
        final_instances = []
        
        # Get unique class IDs
        class_ids = set(inst['class_id'] for inst in self.accumulated_instances)
        
        for class_id in class_ids:
            # Filter instances for this class
            class_instances = [
                inst for inst in self.accumulated_instances
                if inst['class_id'] == class_id
            ]
            
            # Apply NMS
            kept_instances = self._apply_nms(class_instances)
            final_instances.extend(kept_instances)
        
        return final_instances
    
    def _apply_nms(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            instances: List of instances for a single class
            
        Returns:
            List of instances after NMS
        """
        if len(instances) == 0:
            return []
        
        # Extract boxes and scores
        boxes = np.array([inst['bbox'] for inst in instances])  # (N, 4)
        scores = np.array([inst['score'] for inst in instances])  # (N,)
        
        # Sort by score (descending)
        order = np.argsort(-scores)
        
        keep = []
        
        while len(order) > 0:
            # Keep highest scoring box
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes[i], boxes[order[1:]])
            
            # Keep boxes with IoU < threshold
            remaining = np.where(ious < self.nms_threshold)[0]
            order = order[remaining + 1]
        
        # Return kept instances
        return [instances[i] for i in keep]
    
    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU between one box and multiple boxes.
        
        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Multiple boxes (N, 4)
            
        Returns:
            IoU values (N,)
        """
        # Compute intersection
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute areas
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Compute union
        union = box_area + boxes_area - intersection
        
        # Avoid division by zero
        union = np.maximum(union, 1e-6)
        
        return intersection / union
    
    def _create_outputs(self, final_prediction):
        """
        Create output data from final predictions.
        
        
        Converts numpy arrays to Python lists for JSON serialization.
        
        Args:
            final_prediction: List of instance dicts from _finalize_predictions()
            
        Returns:
            Dictionary with:
            - 'instances': list of instance dictionaries (JSON-serializable)
            - 'classes': class information
            - 'output_shape': (height, width)
        """
        # Convert numpy arrays to lists for JSON serialization
        json_instances = []
        for inst in final_prediction:
            json_inst = {
                'class_id': int(inst['class_id']),
                'score': float(inst['score']),
                'bbox': inst['bbox'].tolist() if isinstance(inst['bbox'], np.ndarray) else inst['bbox'],
            }
            
            # Handle mask - could be 'mask', 'mask_local', or missing
            if 'mask' in inst:
                json_inst['mask'] = inst['mask'].tolist() if isinstance(inst['mask'], np.ndarray) else inst['mask']
            elif 'mask_local' in inst:
                json_inst['mask_local'] = inst['mask_local'].tolist() if isinstance(inst['mask_local'], np.ndarray) else inst['mask_local']
                json_inst['mask_offset'] = inst['mask_offset']
            
            # Handle optional fields
            if 'centroid' in inst:
                json_inst['centroid'] = inst['centroid'].tolist() if isinstance(inst['centroid'], np.ndarray) else inst['centroid']
            if 'area' in inst:
                json_inst['area'] = int(inst['area'])
            
            json_instances.append(json_inst)
        
        return {
            'instances': json_instances,
            'classes': self.classes,
            'output_shape': self.output_shape,
            'num_instances': len(json_instances)
        }