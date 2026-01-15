"""
LR Finder - Fast.ai style implementation
Finds optimal learning rate using the Valley method
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Callable
import os

from core.training.visualization import plot_lr_finder
from core.training.config import (
    LR_FINDER_START_LR,
    LR_FINDER_END_LR,
    LR_FINDER_NUM_STEPS,
    LR_FINDER_SMOOTH_FACTOR
)


class LRFinder:
    """
    Learning Rate Finder using exponential range test.
    Uses the Valley method : finds the steepest slope roughly 2/3 
    through the longest valley in the LR plot.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        callback: Optional[Callable] = None
    ):
        """
        Args:
            model: PyTorch model to test
            optimizer: Optimizer instance (will be reset after test)
            criterion: Loss function
            device: torch.device ('cpu' or 'cuda')
            callback: Optional callback(step, total_steps, loss, lr)
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callback = callback
        
        # Results storage
        self.learning_rates = []
        self.losses = []
        
    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = None,
        end_lr: float = None,
        num_steps: int = None,
        smooth_factor: float = None
    ) -> Tuple[float, str]:
        """
        Run LR range test with exponential increase.
        
        Args:
            train_loader: Training DataLoader
            start_lr: Minimum LR to test (default: from config)
            end_lr: Maximum LR to test (default: from config)
            num_steps: Number of steps for the test (default: from config)
            smooth_factor: Smoothing factor for loss (default: from config)
            
        Returns:
            suggested_lr: Best LR found using Valley method
            plot_path: Path to saved plot PNG (None until save_plot is called)
        """
        # Use config defaults if not specified
        start_lr = start_lr if start_lr is not None else LR_FINDER_START_LR
        end_lr = end_lr if end_lr is not None else LR_FINDER_END_LR
        num_steps = num_steps if num_steps is not None else LR_FINDER_NUM_STEPS
        smooth_factor = smooth_factor if smooth_factor is not None else LR_FINDER_SMOOTH_FACTOR

        # Save original state
        original_state = {
            'model': self.model.state_dict().copy(),
            'optimizer': self.optimizer.state_dict().copy()
        }
        
        # Reset storage
        self.learning_rates = []
        self.losses = []
        
        # Calculate LR multiplier (exponential increase)
        lr_mult = (end_lr / start_lr) ** (1 / num_steps)
        
        # Set model to training mode
        self.model.train()
        
        # Initialize LR
        current_lr = start_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Run range test
        step = 0
        data_iter = iter(train_loader)
        smoothed_loss = None
        
        try:
            while step < num_steps:
                # Get batch
                try:
                    images, masks = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, masks = next(data_iter)
                
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Smooth loss (exponential moving average)
                loss_value = loss.item()
                if smoothed_loss is None:
                    smoothed_loss = loss_value
                else:
                    smoothed_loss = smooth_factor * loss_value + (1 - smooth_factor) * smoothed_loss
                
                # Store results
                self.learning_rates.append(current_lr)
                self.losses.append(smoothed_loss)
                
                # Callback for UI update
                if self.callback:
                    self.callback(step + 1, num_steps, smoothed_loss, current_lr)
                
                # Stop if loss explodes (divergence)
                if len(self.losses) > 1 and smoothed_loss > 10 * min(self.losses):
                    break
                
                # Increase LR exponentially
                current_lr *= lr_mult
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                step += 1
                
        except Exception as e:
            # Restore original state on error
            self.model.load_state_dict(original_state['model'])
            self.optimizer.load_state_dict(original_state['optimizer'])
            raise e
        
        # Restore original model/optimizer state
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])
        
        # Calculate suggested LR using Valley method
        suggested_lr = self._find_best_lr_valley()
        
        # Generate plot (will be saved in next step)
        plot_path = None  # Will be set by save_plot()
        
        return suggested_lr, plot_path
    
    def _find_best_lr_valley(self) -> float:
        """
        Valley method : Find the steepest slope roughly 2/3 through 
        the longest valley in the LR plot.
        
        This is the default method used by Fast.ai and recommended for 
        geospatial/real-world data.
        
        Algorithm:
        1. Find the longest decreasing subsequence (valley) in the loss curve
        2. Take a point roughly 2/3 through this valley
        3. Find the steepest gradient around that point
        
        Returns:
            Suggested learning rate
        """
        if len(self.losses) < 10:
            return 0.001  # Fallback if test failed early
        
        losses = np.array(self.losses)
        lrs = np.array(self.learning_rates)
        n = len(losses)
        
        # Skip first 10% and last 5 points (like Fast.ai does)
        skip_start = max(1, n // 10)
        skip_end = max(5, n // 20)
        losses = losses[skip_start:-skip_end]
        lrs = lrs[skip_start:-skip_end]
        n = len(losses)
        
        if n < 10:
            return 0.001
        
        # Find the longest decreasing subsequence (LDS) - this is the "valley"
        # Using dynamic programming
        lds = [1] * n  # Length of longest decreasing subsequence ending at i
        max_start, max_end = 0, 0
        
        for i in range(1, n):
            for j in range(0, i):
                if losses[i] < losses[j] and lds[i] < lds[j] + 1:
                    lds[i] = lds[j] + 1
                    if lds[max_end] < lds[i]:
                        max_end = i
        
        # Reconstruct the valley indices
        valley_length = lds[max_end]
        if valley_length < 5:
            # Valley too short, fallback to simple steepest gradient
            gradients = np.gradient(losses)
            min_grad_idx = np.argmin(gradients)
            suggested_lr = lrs[min_grad_idx]
        else:
            # Find the start of the valley by backtracking
            valley_indices = [max_end]
            current_length = lds[max_end]
            
            for i in range(max_end - 1, -1, -1):
                if lds[i] == current_length - 1 and losses[max_end] < losses[i]:
                    valley_indices.append(i)
                    current_length -= 1
                    max_end = i
                if current_length == 1:
                    break
            
            valley_indices.reverse()
            max_start = valley_indices[0]
            max_end = valley_indices[-1]
            
            # Take a point roughly 2/3 through the valley
            valley_length = max_end - max_start
            target_idx = max_start + int(valley_length * 2 / 3)
            
            # Find steepest gradient in a small window around that point
            window_start = max(max_start, target_idx - 2)
            window_end = min(max_end, target_idx + 3)
            
            if window_end > window_start:
                window_losses = losses[window_start:window_end]
                window_gradients = np.gradient(window_losses)
                local_min_idx = np.argmin(window_gradients)
                suggested_lr = lrs[window_start + local_min_idx]
            else:
                suggested_lr = lrs[target_idx]
        
        # Clamp to reasonable range
        suggested_lr = max(1e-6, min(suggested_lr, 1.0))
        
        return suggested_lr
    
    def save_plot(self, output_dir: str, filename: str = "lr_finder.png") -> str:
        """
        Save LR finder plot to disk.
        
        Uses the visualization module's plot_lr_finder() function.
        
        Args:
            output_dir: Directory to save plot
            filename: Plot filename
            
        Returns:
            Full path to saved plot
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, filename)
        
        # Get suggested LR
        suggested_lr = self._find_best_lr_valley()
        
        # Use visualization module to generate plot
        plot_lr_finder(
            learning_rates=self.learning_rates,
            losses=self.losses,
            suggested_lr=suggested_lr,
            output_path=plot_path
        )
        
        return plot_path