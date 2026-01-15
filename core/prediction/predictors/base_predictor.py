"""
Base Predictor for QPredict

Abstract base class for all predictors.
Defines common interface and shared functionality.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Callable, Optional


class BasePredictor(ABC):
    """
    Abstract base class for predictors.
    
    All task-specific predictors must inherit from this class and implement:
    - _process_batch() - How to run inference on a batch of tiles
    - _accumulate_prediction() - How to accumulate tile predictions
    - _finalize_predictions() - How to merge accumulated predictions
    - _create_outputs() - What outputs to create
    
    Common functionality provided:
    - Tile batch processing
    - Progress tracking
    
    Args:
        model: PyTorch model (already on device and in eval mode)
        batch_size: Number of tiles to process simultaneously
        device: Device where model is located
        model_info: Model information from QMTPLoader
        output_shape: (height, width) of the output
        **kwargs: Task-specific arguments
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        device: torch.device,
        model_info: Dict[str, Any],
        output_shape: tuple,
        **kwargs
    ):
        """
        Initialize base predictor.
        
        Args:
            model: PyTorch model
            batch_size: Batch size for inference
            device: Device (CPU/CUDA)
            model_info: Model configuration
            output_shape: (height, width) of output
            **kwargs: Task-specific arguments
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.model_info = model_info
        self.kwargs = kwargs
        
        # Get model configuration
        self.num_classes = model_info['num_classes']
        self.classes = model_info['classes']
        
        # Output shape
        self.output_shape = output_shape
    
    @abstractmethod
    def _initialize_accumulation(self):
        """
        Initialize arrays for accumulating predictions (task-specific).
        
        Must be implemented by subclasses.
        
        Example (semantic):
            self.prediction_sum = np.zeros((num_classes, H, W), dtype=np.float32)
            self.prediction_count = np.zeros((H, W), dtype=np.uint16)
        
        Example (instance):
            self.accumulated_instances = []
        """
        pass
    
    @abstractmethod
    def _process_batch(self, batch_tiles: list, batch_metadata: list):
        """
        Process a batch of tiles through the model (task-specific).
        
        Must be implemented by subclasses.
        
        Args:
            batch_tiles: List of tile tensors
            batch_metadata: List of tile metadata dicts
        """
        pass
    
    @abstractmethod
    def _accumulate_prediction(self, prediction, metadata: Dict[str, Any]):
        """
        Accumulate a single tile prediction (task-specific).
        
        Must be implemented by subclasses.
        
        Args:
            prediction: Model output (format depends on task)
            metadata: Tile metadata with position info
        """
        pass
    
    @abstractmethod
    def _finalize_predictions(self):
        """
        Finalize predictions by merging accumulated results (task-specific).
        
        Must be implemented by subclasses.
        
        Returns:
            Final prediction in task-specific format
        """
        pass
    
    @abstractmethod
    def _create_outputs(self, final_prediction):
        """
        Create output data from final predictions (task-specific).
        
        Must be implemented by subclasses.
        
        Args:
            final_prediction: Finalized prediction from _finalize_predictions()
            
        Returns:
            Output data (format depends on task)
        """
        pass
    
    def predict(
        self,
        tiles: list,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Run prediction pipeline (common for all tasks).
        
        This method:
        1. Initializes accumulation
        2. Processes tiles in batches
        3. Finalizes predictions
        4. Creates output data
        
        Args:
            tiles: List of tile dictionaries with 'tensor' and metadata
            progress_callback: Optional callback function(current, total)
                              called after each batch
        
        Returns:
            Output data from _create_outputs()
        
        Raises:
            RuntimeError: If prediction fails
        """
        try:
            # Initialize accumulation structures
            self._initialize_accumulation()
            
            # Process tiles in batches
            self._process_tiles_in_batches(tiles, progress_callback)
            
            # Finalize predictions (merge overlaps, etc.)
            final_prediction = self._finalize_predictions()
            
            # Create output data
            output_data = self._create_outputs(final_prediction)
            
            return output_data
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _process_tiles_in_batches(
        self,
        tiles: list,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Process tiles in batches for GPU efficiency (common logic).
        
        Args:
            tiles: List of tile dictionaries
            progress_callback: Progress callback function
        """
        total_tiles = len(tiles)
        
        batch_tiles = []
        batch_metadata = []
        processed_count = 0
        
        for tile_data in tiles:
            batch_tiles.append(tile_data['tensor'])
            batch_metadata.append(tile_data)
            
            # Process batch when full or at end
            if len(batch_tiles) == self.batch_size or processed_count + len(batch_tiles) == total_tiles:
                self._process_batch(batch_tiles, batch_metadata)
                
                processed_count += len(batch_tiles)
                
                # Call progress callback
                if progress_callback:
                    progress_callback(processed_count, total_tiles)
                
                # Clear batch
                batch_tiles = []
                batch_metadata = []