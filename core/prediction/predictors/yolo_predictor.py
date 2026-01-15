"""
YOLO Predictor for QPredict
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

from core.prediction.predictors.base_predictor import BasePredictor


class YOLOPredictor(BasePredictor):
    """
    YOLO predictor with global NMS (server version - FIXED).
    """
    
    def __init__(
        self,
        model,
        batch_size: int,
        device,
        model_info: dict,
        output_shape: tuple,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        overlap: int = 0,
        simplify_tolerance: float = 0.5,
        **kwargs
    ):
        """Initialize YOLO predictor."""
        super().__init__(
            model=model,
            batch_size=batch_size,
            device=device,
            model_info=model_info,
            output_shape=output_shape,
            **kwargs
        )
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.overlap = overlap
        self.simplify_tolerance = simplify_tolerance
        
        # Extract YOLO task from model
        self.yolo_task = getattr(model, '_qpredict_yolo_task', 'detect')
        
        # Storage for all detections across tiles
        self.all_detections = []

        # Configure device for YOLO inference
        if isinstance(device, str):
            if device == 'cpu':
                self.yolo_device = 'cpu'
            elif 'cuda' in device:
                # Extract index if present (cuda:0 → 0)
                try:
                    idx = int(device.split(':')[1]) if ':' in device else 0
                    self.yolo_device = idx
                except:
                    self.yolo_device = 0
            else:
                self.yolo_device = device
        elif hasattr(device, 'type'):
            # device is torch.device
            if device.type == 'cpu':
                self.yolo_device = 'cpu'
            else:
                self.yolo_device = device.index if device.index is not None else 0
        else:
            self.yolo_device = 'cpu'
        
        print(f"[YOLOPredictor] YOLO device configured: {self.yolo_device}")

    def _initialize_accumulation(self):
        """Initialize detection storage."""
        self.all_detections = []

    def _process_batch(self, batch_tiles, batch_metadata):
        """
        Process a batch of tiles through YOLO.
        
        Args:
            batch_tiles: List of tile tensors (C, H, W) - NORMALIZED by RasterProcessor
            batch_metadata: List of tile metadata dicts
        """
        for tile_tensor, metadata in zip(batch_tiles, batch_metadata):
            self._process_single_tile(tile_tensor, metadata)
    
    def _process_single_tile(self, tile_tensor, metadata):
        """
        Process a single tile through YOLO.
        
        CRITICAL: YOLO expects images in [0, 255] range (or [0, 1])
        The tile_tensor comes normalized from RasterProcessor.
        We need to DENORMALIZE it back to [0, 255].
        
        Args:
            tile_tensor: Tile tensor (C, H, W) - NORMALIZED
            metadata: Tile metadata with position info
        """
        # Get tile coordinates
        x_min = metadata['x_min']
        y_min = metadata['y_min']
        x_max = metadata['x_max']
        y_max = metadata['y_max']
        tile_width = metadata['tile_width']
        tile_height = metadata['tile_height']
        
        # ================================================================
        # Convert tile to format expected by YOLO
        # ================================================================
        # Tile is (C, H, W) in [0, 1] range (passthrough normalization from QMTP)
        # YOLO expects uint8 in [0, 255] range
        
        tile_np = tile_tensor.cpu().numpy()  # (C, H, W)
        
        # Scale to [0, 255] and convert to uint8
        tile_np = np.clip(tile_np * 255.0, 0, 255).astype(np.uint8)
        
        # Transpose to HWC for YOLO (expects HWC format)
        tile_np = np.transpose(tile_np, (1, 2, 0))  # (H, W, C)
        
        # ================================================================
        # Run YOLO inference
        # ================================================================
        # Note: YOLO will normalize internally ([0, 255] → [0, 1])
        results = self.model.predict(
            tile_np,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.yolo_device,
            verbose=False,
            # Important: disable augmentation during inference
            augment=False,
            # Use half precision if available (faster)
            half=False  # Set to True if you want FP16 (requires CUDA)
        )
        
        # Extract detections from YOLO Results
        if results is None or len(results) == 0:
            return  # No detections
        
        result = results[0]
        
        # Check if result has detections
        if result is None:
            return  # No valid result
        
        # Check based on task type
        if self.yolo_task == 'obb':
            if not hasattr(result, 'obb') or result.obb is None or len(result.obb) == 0:
                return  # No OBB detections
        elif self.yolo_task == 'segment':
            if not hasattr(result, 'masks') or result.masks is None or len(result.boxes) == 0:
                return  # No segment detections
        else:  # detect
            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                return  # No detections
        
        # Process based on YOLO task
        if self.yolo_task == 'segment':
            self._process_segment_detections(result, x_min, y_min, tile_width, tile_height)
        elif self.yolo_task == 'obb':
            self._process_obb_detections(result, x_min, y_min, tile_width, tile_height)
        else:  # detect
            self._process_detect_detections(result, x_min, y_min, tile_width, tile_height)
    
    def _process_detect_detections(self, result, x_min, y_min, tile_width, tile_height):
        """Process YOLO detection results."""
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()  # (N,)
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        
        margin = max(32, self.overlap // 4)
        
        for box, score, cls in zip(boxes, scores, classes):
            # Centroid filtering
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            if not (margin <= cx < tile_width - margin and
                    margin <= cy < tile_height - margin):
                continue
            
            # Convert to global coordinates
            global_box = [
                float(box[0] + x_min),
                float(box[1] + y_min),
                float(box[2] + x_min),
                float(box[3] + y_min)
            ]
            
            detection = {
                'type': 'bbox',
                'bbox': global_box,
                'class_id': int(cls),
                'class_name': self.classes[cls]['name'],
                'score': float(score)
            }
            
            self.all_detections.append(detection)
    
    def _process_obb_detections(self, result, x_min, y_min, tile_width, tile_height):
        """Process YOLO OBB results."""
        if not hasattr(result, 'obb') or result.obb is None:
            return
        
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
        scores = result.obb.conf.cpu().numpy()  # (N,)
        classes = result.obb.cls.cpu().numpy().astype(int)  # (N,)
        
        margin = max(32, self.overlap // 4)
        
        for corners, score, cls in zip(xyxyxyxy, scores, classes):
            # Centroid filtering
            cx = corners[:, 0].mean()
            cy = corners[:, 1].mean()
            
            if not (margin <= cx < tile_width - margin and
                    margin <= cy < tile_height - margin):
                continue
            
            # Convert to global coordinates
            global_corners = corners + np.array([x_min, y_min])
            
            detection = {
                'type': 'obb',
                'corners': global_corners.tolist(),
                'class_id': int(cls),
                'class_name': self.classes[cls]['name'],
                'score': float(score)
            }
            
            self.all_detections.append(detection)
    
    def _process_segment_detections(self, result, x_min, y_min, tile_width, tile_height):
        """Process YOLO segmentation results."""
        if not hasattr(result, 'masks') or result.masks is None:
            return
        
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.xy  # List of polygons (N arrays of shape (M, 2))
        
        margin = max(32, self.overlap // 4)
        
        for box, score, cls, mask_coords in zip(boxes, scores, classes, masks):
            # Centroid filtering
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
            if not (margin <= cx < tile_width - margin and
                    margin <= cy < tile_height - margin):
                continue
            
            # Convert mask to global coordinates
            global_mask = mask_coords + np.array([x_min, y_min])
            
            detection = {
                'type': 'polygon',
                'polygon': global_mask.tolist(),
                'bbox': [float(x) for x in (box + np.array([x_min, y_min, x_min, y_min]))],
                'class_id': int(cls),
                'class_name': self.classes[cls]['name'],
                'score': float(score)
            }
            
            self.all_detections.append(detection)
    
    def _accumulate_prediction(self, prediction, metadata):
        """Not used by YOLO predictor (processes tiles directly)."""
        pass
    
    def _finalize_predictions(self):
        """
        Apply global NMS to all detections.
        
        Returns:
            List of final detections after NMS
        """
        if not self.all_detections:
            return []
        
        print(f"[YOLOPredictor] Total detections before NMS: {len(self.all_detections)}")
        
        # Apply NMS per class
        final_detections = []
        
        # Get unique class IDs
        class_ids = set(det['class_id'] for det in self.all_detections)
        
        for class_id in class_ids:
            # Filter detections for this class
            class_detections = [
                det for det in self.all_detections
                if det['class_id'] == class_id
            ]
            
            print(f"[YOLOPredictor] Class {class_id}: {len(class_detections)} detections before NMS")
            
            # Apply NMS
            kept_detections = self._apply_nms(class_detections)
            
            print(f"[YOLOPredictor] Class {class_id}: {len(kept_detections)} detections after NMS")
            
            final_detections.extend(kept_detections)
        
        print(f"[YOLOPredictor] Total detections after NMS: {len(final_detections)}")
        
        return final_detections
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression.
        
        Args:
            detections: List of detections for a single class
            
        Returns:
            List of detections after NMS
        """
        if len(detections) == 0:
            return []
        
        # Extract boxes and scores
        boxes = []
        for det in detections:
            if det['type'] == 'obb':
                # For OBB, compute axis-aligned bbox from corners
                corners = np.array(det['corners'])
                box = [
                    corners[:, 0].min(),
                    corners[:, 1].min(),
                    corners[:, 0].max(),
                    corners[:, 1].max()
                ]
            else:
                box = det['bbox']
            boxes.append(box)
        
        boxes = np.array(boxes)
        scores = np.array([det['score'] for det in detections])
        
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
            ious = self._compute_iou_batch(boxes[i], boxes[order[1:]])
            
            # Keep boxes with IoU < threshold
            remaining = np.where(ious < self.iou_threshold)[0]
            order = order[remaining + 1]
        
        return [detections[i] for i in keep]
    
    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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
        union = np.maximum(union, 1e-6)
        
        return intersection / union
    
    def _create_outputs(self, final_prediction):
        """
        Create output data from final predictions.
        
        SERVER VERSION: Returns detection data (no file creation).
        
        Args:
            final_prediction: List of detection dicts from _finalize_predictions()
            
        Returns:
            Dictionary with:
            - 'detections': list of detection dictionaries
            - 'yolo_task': task type (detect, segment, obb)
            - 'classes': class information
            - 'num_detections': total count
        """
        return {
            'detections': final_prediction,
            'yolo_task': self.yolo_task,
            'classes': self.classes,
            'num_detections': len(final_prediction)
        }