"""
Semantic Segmentation Predictor for QPredict

Handles inference for semantic segmentation models (UNet, FPN, DeepLabV3, etc.)
with pixel-wise classification and overlap averaging.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional

from core.prediction.predictors.base_predictor import BasePredictor
from core.prediction.reliability_maps import ReliabilityMapCalculator


class SemanticSegmentationPredictor(BasePredictor):
    """
    Predictor for semantic segmentation models (server version).
    
    Handles:
    - Batch inference with logits output
    - Padding removal from predictions
    - Overlap averaging (logits → softmax → argmax)
    - Reliability maps (confidence/uncertainty)
    
    Returns numpy arrays (no GeoTIFF creation).
    
    Args:
        model: PyTorch model (semantic segmentation)
        batch_size: Batch size for inference
        device: Device (CPU/CUDA)
        model_info: Model configuration
        output_shape: (height, width) of the output
        apply_colors: Whether to include color info in output (default: True)
        output_format: 'class_raster' or 'reliability_map' (default: 'class_raster')
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize semantic segmentation predictor."""
        super().__init__(*args, **kwargs)
        
        # Extract semantic-specific arguments
        self.apply_colors = self.kwargs.get('apply_colors', True)
        self.output_format = self.kwargs.get('output_format', 'class_raster')
        
        # Initialize reliability calculator if needed
        self.reliability_calculator = None
        self.reliability_type = self.model_info.get('reliability_type')
        if self.output_format == 'reliability_map':
            self.reliability_calculator = ReliabilityMapCalculator(self.num_classes)
    
    def _initialize_accumulation(self):
        """
        Initialize arrays for accumulating predictions.
        
        For semantic segmentation:
        - prediction_sum: Accumulates logits (for averaging)
        - prediction_count: Counts overlaps per pixel
        """
        # For vector output, we need to create a class_raster first
        actual_format = 'class_raster' if self.output_format == 'vector' else self.output_format
        
        if actual_format == 'class_raster':
            # Class raster: accumulate logits
            self.prediction_sum = np.zeros(
                (self.num_classes, *self.output_shape),
                dtype=np.float32
            )
        elif actual_format == 'reliability_map':
            # Reliability map: accumulate metrics (computed per-tile)
            self.prediction_sum = np.zeros(
                self.output_shape,
                dtype=np.float32
            )
        
        self.prediction_count = np.zeros(self.output_shape, dtype=np.uint16)
    
    def _process_batch(self, batch_tiles: list, batch_metadata: list):
        """
        Process a batch of tiles through the model.
        
        Handles tiles of different sizes (at raster edges) by processing
        them individually when batch contains mixed sizes.
        
        Args:
            batch_tiles: List of tile tensors
            batch_metadata: List of tile metadata dicts
        """
        # Check if all tiles have same shape
        shapes = [tuple(t.shape) for t in batch_tiles]
        all_same_shape = len(set(shapes)) == 1
        
        if all_same_shape:
            # Fast path: stack and process as batch
            batch_tensor = torch.stack(batch_tiles).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(batch_tensor)
            
            # Move predictions to CPU
            predictions = predictions.cpu()
            
            # Process each prediction
            for pred, metadata in zip(predictions, batch_metadata):
                self._accumulate_prediction(pred, metadata)
        
        else:
            # Mixed sizes: process tiles individually
            for tile_tensor, metadata in zip(batch_tiles, batch_metadata):
                # Add batch dimension
                tile_batch = tile_tensor.unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    pred_batch = self.model(tile_batch)
                
                # Remove batch dimension and move to CPU
                pred = pred_batch.squeeze(0).cpu()
                
                # Process prediction
                self._accumulate_prediction(pred, metadata)
    
    def _accumulate_prediction(self, prediction: torch.Tensor, metadata: Dict[str, Any]):
        """
        Accumulate a single tile prediction into the output array.
        
        Handles:
        - Padding removal (crop central region with asymmetric padding support)
        - Overlap accumulation for averaging
        
        Args:
            prediction: Model output tensor of shape (C, H_padded, W_padded)
            metadata: Tile metadata with position info
                Required keys:
                - tile_width, tile_height: actual tile dimensions
                - pad_left, pad_top: padding applied
                - out_x, out_y: position in output array
        """
        # Get tile position (without padding)
        tile_width = metadata['tile_width']
        tile_height = metadata['tile_height']
        
        # Get actual padding from metadata
        pad_left = metadata['pad_left']
        pad_top = metadata['pad_top']
        
        # Crop to remove padding (including divisibility padding)
        pred_height = prediction.shape[1]
        pred_width = prediction.shape[2]
        
        # Total padding includes both edge padding and divisibility padding
        total_pad_height = pred_height - tile_height
        total_pad_width = pred_width - tile_width
        
        if total_pad_height > 0 or total_pad_width > 0:
            # Crop from pad_top to pad_top + tile_height
            # This removes both edge padding and divisibility padding
            pred_cropped = prediction[
                :,
                pad_top:pad_top + tile_height,
                pad_left:pad_left + tile_width
            ]
        else:
            pred_cropped = prediction[:, :tile_height, :tile_width]
        
        # Convert to numpy
        pred_np = pred_cropped.numpy()  # Shape: (C, H, W)

        # Get position in output array (from metadata)
        out_y = metadata['out_y']
        out_x = metadata['out_x']

        # Accumulate prediction
        actual_format = 'class_raster' if self.output_format == 'vector' else self.output_format
        
        if actual_format == 'reliability_map':
            # For reliability maps: compute metric before accumulating
            if self.reliability_type and 'Confidence' in self.reliability_type:
                tile_reliability = self.reliability_calculator.compute_confidence(pred_np)
            else:
                tile_reliability = self.reliability_calculator.compute_uncertainty(
                    pred_np,
                    normalized=False
                )
            
            # Accumulate the reliability metric
            self.prediction_sum[
                out_y:out_y + tile_height,
                out_x:out_x + tile_width
            ] += tile_reliability
        else:
            # For class_raster: accumulate logits
            self.prediction_sum[
                :,
                out_y:out_y + tile_height,
                out_x:out_x + tile_width
            ] += pred_np
        
        # Count overlaps
        self.prediction_count[
            out_y:out_y + tile_height,
            out_x:out_x + tile_width
        ] += 1
    
    def _finalize_predictions(self):
        """
        Finalize predictions by averaging overlaps.
        
        Returns:
            Final prediction array:
            - class_raster/vector: (H, W) uint8 with class IDs
            - reliability_map: (H, W) float32 with metric values
        """
        # For vector output, we create a class_raster
        actual_format = 'class_raster' if self.output_format == 'vector' else self.output_format
        
        if actual_format == 'class_raster':
            # Average logits where overlaps exist
            mask = self.prediction_count > 0
            averaged_logits = np.zeros_like(self.prediction_sum)
            averaged_logits[:, mask] = (
                self.prediction_sum[:, mask] / self.prediction_count[mask]
            )
            
            # Apply softmax and argmax to get class predictions
            from scipy.special import softmax
            probs = softmax(averaged_logits, axis=0)
            prediction = np.argmax(probs, axis=0).astype(np.uint8)
            
            return prediction
        
        elif actual_format == 'reliability_map':
            # Average reliability metric
            mask = self.prediction_count > 0
            averaged_reliability = np.zeros(self.output_shape, dtype=np.float32)
            averaged_reliability[mask] = (
                self.prediction_sum[mask] / self.prediction_count[mask]
            )
            
            return averaged_reliability
    
    def _create_outputs(self, final_prediction):
        """
        Create output data from final predictions.
        
        SERVER VERSION: Returns numpy array + metadata (no file creation).
        
        Args:
            final_prediction: Finalized prediction from _finalize_predictions()
            
        Returns:
            Dictionary with:
            - 'prediction': numpy array
            - 'output_format': format type
            - 'classes': class information (if class_raster/vector)
            - 'reliability_type': type of metric (if reliability_map)
        """
        output_data = {
            'prediction': final_prediction,
            'output_format': self.output_format,
        }
        
        if self.output_format in ['class_raster', 'vector']:
            # Include class information for color tables / vectorization
            output_data['classes'] = self.classes
            output_data['apply_colors'] = self.apply_colors
        
        elif self.output_format == 'reliability_map':
            # Include reliability type for interpretation
            output_data['reliability_type'] = self.reliability_type
            output_data['num_classes'] = self.num_classes
        
        return output_data