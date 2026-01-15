"""
Training engine for QModel Trainer

This module handles the core training loop for deep learning models.
Supports semantic segmentation with PyTorch and provides callbacks
for progress monitoring and logging.

"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Import metrics from new module
from core.training.metrics import (
    compute_iou,
    compute_f1_score,
    compute_precision_recall,
    compute_pixel_accuracy
)

try:
    from torchvision.ops import box_iou
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class SemanticSegmentationLoss(nn.Module):
    """
    Combined loss for semantic segmentation.
    
    Uses CrossEntropyLoss which combines LogSoftmax and NLLLoss.
    Can be extended to include additional losses (Dice, Focal, etc.)
    
    Args:
        ignore_index: Index to ignore in loss computation (e.g., for padding)
        weight: Class weights for imbalanced datasets
    """
    
    def __init__(self, ignore_index: int = -100, weight: Optional[torch.Tensor] = None):
        """Initialize the loss function."""
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model output (B, C, H, W) - logits
            targets: Ground truth masks (B, H, W) - class indices
            
        Returns:
            Loss value
        """
        return self.criterion(predictions, targets)


# =============================================================================
# SEMANTIC SEGMENTATION TRAINER
# =============================================================================

class SemanticSegmentationTrainer:
    """
    Trainer for semantic segmentation models.
    
    Handles the complete training loop including:
    - Training and validation epochs
    - Metric computation (IoU, F1, accuracy)
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Progress callbacks for UI updates
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ('cuda' or 'cpu')
        num_classes: Number of segmentation classes
        output_dir: Directory to save checkpoints and logs
        callbacks: Dictionary of callback functions for progress updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        num_classes: int = 2,
        architecture: Optional[str] = None,      
        encoder_name: Optional[str] = None,     
        output_dir: str = './outputs',
        callbacks: Optional[Dict[str, Callable]] = None,
        save_best: bool = True 
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.architecture = architecture           
        self.encoder_name = encoder_name           
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or {}
        self.save_best = save_best
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = SemanticSegmentationLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_iou = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.patience_counter = 0
        
        # Stop flag
        self.should_stop = False
        
        # Training time tracking (Total training time in seconds)
        self.total_train_time = 0.0  
    
    def stop(self):
        """Request to stop training."""
        self.should_stop = True

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Callback for batch progress
            if 'on_batch_end' in self.callbacks:
                self.callbacks['on_batch_end'](
                    epoch=epoch,
                    batch=batch_idx,
                    total_batches=num_batches,
                    loss=loss.item()
                )
        
        # Average loss
        avg_loss = total_loss / num_batches
        
        return avg_loss
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Compute metrics
                iou = compute_iou(outputs, masks, self.num_classes)
                f1 = compute_f1_score(outputs, masks, self.num_classes)
                precision, recall = compute_precision_recall(outputs, masks, self.num_classes)
                
                # Accumulate
                total_loss += loss.item()
                total_iou += iou
                total_f1 += f1
                total_precision += precision
                total_recall += recall
        
        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'iou': total_iou / num_batches,
            'f1': total_f1 / num_batches,
            'precision': total_precision / num_batches,
            'recall': total_recall / num_batches
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None,
        architecture: Optional[str] = None,
        encoder_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        in_channels: int = 3
    ):
        """
        Save a training checkpoint with metadata for resume training.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
            architecture: Model architecture (e.g., 'UNet')
            encoder_name: Encoder/backbone name (e.g., 'resnet50')
            num_classes: Number of output classes
            in_channels: Number of input channels (default: 3 for RGB)
        """
        # Only save if save_best is True (or if it's not a best model checkpoint)
        if is_best and not self.save_best:
            return  # Don't save best model if save_best=False
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
            'history': self.history,
            # Metadata for resume training validation
            'metadata': {
                'architecture': architecture,
                'encoder_name': encoder_name,
                'num_classes': num_classes or self.num_classes,
                'in_channels': in_channels
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if filename is None:
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # ALWAYS save model weights when saving a checkpoint (for QMTP export)
        if is_best:
            model_path = self.output_dir / 'best_model_weights.pth'
            torch.save(self.model.state_dict(), model_path)

    def load_checkpoint_state(self, checkpoint: Dict[str, Any]) -> int:
        """
        Load checkpoint state into trainer (called after external validation).
        
        This method assumes the checkpoint has already been validated
        by checkpoint_loader.load_checkpoint_for_resume().
        
        Args:
            checkpoint: Validated checkpoint dictionary
            
        Returns:
            Next epoch to start from (checkpoint['epoch'] + 1)
        """
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if present
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.best_iou = checkpoint.get('best_iou', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        # Return next epoch to start from
        return checkpoint.get('epoch', 0) + 1
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = True):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: If True, restore optimizer and scheduler states
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if present
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore training state
            self.best_iou = checkpoint.get('best_iou', 0.0)
            self.history = checkpoint.get('history', self.history)
            # Return next epoch to start from
            return checkpoint.get('epoch', 0) + 1  
        
        return 0
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (None = disabled)
            save_best_only: If True, only save best model. If False, save every 10 epochs
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_epoch = 0

        # Start global timer
        global_start_time = time.time()
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            # Import checkpoint loader
            from .checkpoint_loader import load_checkpoint_for_resume
            
            # Load and validate checkpoint
            success, checkpoint, error = load_checkpoint_for_resume(
                checkpoint_path=resume_from,
                config={
                    'architecture': self.architecture,
                    'backbone': self.encoder_name,
                },
                device=torch.device(self.device),
                log_callback=self.callbacks.get('on_log')
            )
            
            if not success:
                raise RuntimeError(f"Failed to load checkpoint: {error}")
            
            # Load checkpoint state into trainer
            start_epoch = self.load_checkpoint_state(checkpoint)
            
            if 'on_log' in self.callbacks:
                self.callbacks['on_log'](
                    f"\n🔄 Resuming training from epoch {start_epoch + 1}/{num_epochs}"
                )
        
        # Callback for training start
        if 'on_train_start' in self.callbacks:
            self.callbacks['on_train_start'](
                total_epochs=num_epochs,
                start_epoch=start_epoch
            )
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Callback for epoch start
            if 'on_epoch_start' in self.callbacks:
                self.callbacks['on_epoch_start'](epoch=epoch)
            
            # Train one epoch
            train_loss = self.train_one_epoch(epoch)

            # CHECK STOP FLAG AGAIN (NEW)
            if self.should_stop:
                if 'on_log' in self.callbacks:
                    self.callbacks['on_log']("\n🛑 Training stopped by user")
                break
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision']) 
            self.history['val_recall'].append(val_metrics['recall'])       
            self.history['learning_rate'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Check if best model
            is_best = val_metrics['iou'] > self.best_iou
            if is_best:
                self.best_iou = val_metrics['iou']
                self.best_epoch = epoch
                self.patience_counter = 0
                if self.save_best:
                    self.save_checkpoint(
                        epoch=epoch,
                        is_best=True,
                        architecture=self.architecture,
                        encoder_name=self.encoder_name,
                        num_classes=self.num_classes,
                        in_channels=3
                    )
            else:
                self.patience_counter += 1
            
            # Callback for epoch end
            if 'on_epoch_end' in self.callbacks:
                self.callbacks['on_epoch_end'](
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics['loss'],
                    val_iou=val_metrics['iou'],  
                    val_ap=None,  # Not used for Semantic Seg
                    val_f1=val_metrics['f1'],
                    val_precision=val_metrics['precision'],
                    val_recall=val_metrics['recall'],
                    learning_rate=current_lr,
                    epoch_time=epoch_time,
                    is_best=is_best
                )
            
            # Early stopping check
            if early_stopping_patience is not None:
                if self.patience_counter >= early_stopping_patience:
                    if 'on_log' in self.callbacks:
                        self.callbacks['on_log'](
                            f"\nEarly stopping triggered after {early_stopping_patience} "
                            f"epochs without improvement.\n"
                            f"Best IoU: {self.best_iou:.4f} at epoch {self.best_epoch + 1}"
                        )
                    break
        
        # Save final model (if training wasn't stopped prematurely)
        if not self.should_stop:
            self.save_checkpoint(num_epochs - 1, is_best=False, filename='final_model.pth')
        
        # Calculate total training time
        self.total_train_time = time.time() - global_start_time
        
        # Save training history (with total time)
        history_path = self.output_dir / 'training_history.pth'
        history_with_metadata = {
            'history': self.history,
            'total_train_time': self.total_train_time,
            'best_iou': self.best_iou,
            'best_epoch': self.best_epoch
        }
        torch.save(history_with_metadata, history_path)
        
        # Save training summary
        self._save_training_summary(num_epochs)
        
        # Callback for training end
        if 'on_train_end' in self.callbacks:
            self.callbacks['on_train_end'](
                best_iou=self.best_iou,
                best_epoch=self.best_epoch
            )
    
    def _save_training_summary(self, num_epochs: int):
        """
        Save a text summary of training.
        
        Args:
            num_epochs: Total number of epochs trained
        """
        summary_path = self.output_dir / 'training_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {num_epochs}\n")
            
            # Format training time
            hours = int(self.total_train_time // 3600)
            minutes = int((self.total_train_time % 3600) // 60)
            seconds = int(self.total_train_time % 60)
            
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            f.write(f"Total training time: {time_str}\n\n")
            
            if self.history['val_iou']:
                f.write(f"Final validation IoU: {self.history['val_iou'][-1]:.4f}\n")
                f.write(f"Final validation F1: {self.history['val_f1'][-1]:.4f}\n")
                f.write(f"Final validation Precision: {self.history['val_precision'][-1]:.4f}\n")
                f.write(f"Final validation Recall: {self.history['val_recall'][-1]:.4f}\n")      
                f.write(f"Final validation loss: {self.history['val_loss'][-1]:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")


# =============================================================================
# INSTANCE SEGMENTATION TRAINER
# =============================================================================

class InstanceSegmentationTrainer:
    """
    Trainer for instance segmentation models (Mask R-CNN).
    
    Handles the complete training loop including:
    - Training and validation epochs
    - Metric computation (AP, AR, Precision, Recall, F1)
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Progress callbacks for UI updates
    
    Args:
        model: Mask R-CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ('cuda' or 'cpu')
        num_classes: Number of classes (INCLUDING background)
        architecture: Architecture name (e.g., 'Mask R-CNN')
        encoder_name: Encoder backbone name (e.g., 'resnet50_fpn')
        output_dir: Directory to save checkpoints and logs
        callbacks: Dictionary of callback functions for progress updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        num_classes: int = 2,
        architecture: Optional[str] = None,
        encoder_name: Optional[str] = None,
        output_dir: str = './outputs',
        callbacks: Optional[Dict[str, Callable]] = None,
        save_best: bool = True 
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.output_dir = Path(output_dir)
        self.callbacks = callbacks or {}
        self.save_best = save_best
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_loss_classifier': [],
            'train_loss_box_reg': [],
            'train_loss_mask': [],
            'train_loss_objectness': [],
            'train_loss_rpn_box_reg': [],
            'val_loss': [],
            'val_ap': [],
            'val_ar': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        # Best model tracking (based on AP)
        self.best_ap = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.patience_counter = 0
        
        # Stop flag
        self.should_stop = False
        
        # Training time tracking
        self.total_train_time = 0.0
    
    def stop(self):
        """Request to stop training."""
        self.should_stop = True
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'loss_classifier': 0.0,
            'loss_box_reg': 0.0,
            'loss_mask': 0.0,
            'loss_objectness': 0.0,
            'loss_rpn_box_reg': 0.0
        }
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass - Mask R-CNN returns loss dict in training mode
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            # Sum all losses
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            self.optimizer.step()

            # Step OneCycleLR scheduler per batch (not per epoch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate losses
            total_loss += losses.item()
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value.item()
            
            # Callback for batch progress
            if 'on_batch_end' in self.callbacks:
                self.callbacks['on_batch_end'](
                    epoch=epoch,
                    batch=batch_idx,
                    total_batches=num_batches,
                    loss=losses.item()
                )
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in loss_components.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_components
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        from core.training.metrics.segmentation import (
            compute_instance_ap,
            compute_instance_ar,
            compute_instance_precision_recall_f1
        )
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets_device = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = self.model(images)
                
                # Store for metrics computation
                all_predictions.extend(predictions)
                all_targets.extend(targets_device)
                
                # Compute loss (put model in train mode temporarily)
                self.model.train()
                loss_dict = self.model(images, targets_device)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1
                self.model.eval()
        
        # Average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute metrics
        ap = compute_instance_ap(all_predictions, all_targets, self.num_classes)
        ar = compute_instance_ar(all_predictions, all_targets, self.num_classes)
        precision, recall, f1 = compute_instance_precision_recall_f1(
            all_predictions, all_targets, self.num_classes
        )
        
        metrics = {
            'loss': avg_loss,
            'ap': ap,
            'ar': ar,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save_checkpoint(
            self,
            epoch: int,
            is_best: bool = False,
            filename: Optional[str] = None,
            architecture: Optional[str] = None,
            encoder_name: Optional[str] = None,
            num_classes: Optional[int] = None,
            in_channels: int = 3
        ):
        """
        Save a training checkpoint with metadata for resume training.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
            architecture: Model architecture
            encoder_name: Encoder/backbone name
            num_classes: Number of output classes
            in_channels: Number of input channels
        """
        # Only save if save_best is True (or if it's not a best model checkpoint)
        if is_best and not self.save_best:
            return  # Don't save best model if save_best=False
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_ap': self.best_ap,
            'history': self.history,
            'metadata': {
                'architecture': architecture,
                'encoder_name': encoder_name,
                'num_classes': num_classes or self.num_classes,
                'in_channels': in_channels
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if filename is None:
            filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # ALWAYS save model weights when saving a checkpoint (for QMTP export)
        if is_best:
            model_path = self.output_dir / 'best_model_weights.pth'
            torch.save(self.model.state_dict(), model_path)
    
    def load_checkpoint_state(self, checkpoint: Dict[str, Any]) -> int:
        """
        Load checkpoint state into trainer.
        
        Args:
            checkpoint: Validated checkpoint dictionary
            
        Returns:
            Next epoch to start from
        """
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if present
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.best_ap = checkpoint.get('best_ap', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        # Return next epoch to start from
        return checkpoint.get('epoch', 0) + 1
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        # Start global timer
        global_start_time = time.time()
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            from .checkpoint_loader import load_checkpoint_for_resume
            
            success, checkpoint, error = load_checkpoint_for_resume(
                checkpoint_path=resume_from,
                config={
                    'architecture': self.architecture,
                    'backbone': self.encoder_name,
                },
                device=torch.device(self.device),
                log_callback=self.callbacks.get('on_log')
            )
            
            if not success:
                raise RuntimeError(f"Failed to load checkpoint: {error}")
            
            start_epoch = self.load_checkpoint_state(checkpoint)
            
            if 'on_log' in self.callbacks:
                self.callbacks['on_log'](
                    f"\n🔄 Resuming training from epoch {start_epoch + 1}/{num_epochs}"
                )
        
        # Callback for training start
        if 'on_train_start' in self.callbacks:
            self.callbacks['on_train_start'](
                total_epochs=num_epochs,
                start_epoch=start_epoch
            )
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Callback for epoch start
            if 'on_epoch_start' in self.callbacks:
                self.callbacks['on_epoch_start'](epoch=epoch)
            
            # Train one epoch
            train_losses = self.train_one_epoch(epoch)
            
            # Check stop flag
            if self.should_stop:
                if 'on_log' in self.callbacks:
                    self.callbacks['on_log']("\n🛑 Training stopped by user")
                break
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate (OneCycleLR is already stepped per batch in train_one_epoch)
            # So we skip the per-epoch step here
            pass
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['train_loss_classifier'].append(train_losses['loss_classifier'])
            self.history['train_loss_box_reg'].append(train_losses['loss_box_reg'])
            self.history['train_loss_mask'].append(train_losses['loss_mask'])
            self.history['train_loss_objectness'].append(train_losses['loss_objectness'])
            self.history['train_loss_rpn_box_reg'].append(train_losses['loss_rpn_box_reg'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_ap'].append(val_metrics['ap'])
            self.history['val_ar'].append(val_metrics['ar'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Check if best model (based on AP)
            is_best = val_metrics['ap'] > self.best_ap
            if is_best:
                self.best_ap = val_metrics['ap']
                self.best_epoch = epoch
                self.patience_counter = 0
                if self.save_best:
                    self.save_checkpoint(
                        epoch=epoch,
                        is_best=True,
                        architecture=self.architecture,
                        encoder_name=self.encoder_name,
                        num_classes=self.num_classes,
                        in_channels=3
                    )
            else:
                self.patience_counter += 1
            
            # Callback for epoch end
            if 'on_epoch_end' in self.callbacks:
                self.callbacks['on_epoch_end'](
                    epoch=epoch,
                    train_loss=train_losses['total_loss'],
                    val_loss=val_metrics['loss'],
                    val_iou=None,  # Not used for instance segmentation
                    val_ap=val_metrics['ap'], 
                    val_f1=val_metrics['f1'],
                    val_precision=val_metrics['precision'],
                    val_recall=val_metrics['recall'],
                    learning_rate=current_lr,
                    epoch_time=epoch_time,
                    is_best=is_best
                )
            
            # Early stopping check
            if early_stopping_patience is not None:
                if self.patience_counter >= early_stopping_patience:
                    if 'on_log' in self.callbacks:
                        self.callbacks['on_log'](
                            f"\nEarly stopping triggered after {early_stopping_patience} "
                            f"epochs without improvement.\n"
                            f"Best AP: {self.best_ap:.4f} at epoch {self.best_epoch + 1}"
                        )
                    break
        
        # Save final model
        if not self.should_stop:
            self.save_checkpoint(num_epochs - 1, is_best=False, filename='final_model.pth')
        
        # Calculate total training time
        self.total_train_time = time.time() - global_start_time
        
        # Save training history
        history_path = self.output_dir / 'training_history.pth'
        history_with_metadata = {
            'history': self.history,
            'total_train_time': self.total_train_time,
            'best_ap': self.best_ap,
            'best_epoch': self.best_epoch
        }
        torch.save(history_with_metadata, history_path)
        
        # Save training summary
        self._save_training_summary(num_epochs)
        
        # Callback for training end
        if 'on_train_end' in self.callbacks:
            self.callbacks['on_train_end'](
                best_iou=self.best_ap,  # Use AP as primary metric
                best_epoch=self.best_epoch
            )
    
    def _save_training_summary(self, num_epochs: int):
        """
        Save a text summary of training.
        
        Args:
            num_epochs: Total number of epochs trained
        """
        summary_path = self.output_dir / 'training_summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("TRAINING SUMMARY - INSTANCE SEGMENTATION\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {num_epochs}\n")
            
            # Format training time
            hours = int(self.total_train_time // 3600)
            minutes = int((self.total_train_time % 3600) // 60)
            seconds = int(self.total_train_time % 60)
            
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            
            f.write(f"Total training time: {time_str}\n\n")
            
            if self.history['val_ap']:
                f.write(f"Best validation AP: {self.best_ap:.4f} at epoch {self.best_epoch + 1}\n")
                f.write(f"Final validation AP: {self.history['val_ap'][-1]:.4f}\n")
                f.write(f"Final validation AR: {self.history['val_ar'][-1]:.4f}\n")
                f.write(f"Final validation F1: {self.history['val_f1'][-1]:.4f}\n")
                f.write(f"Final validation Precision: {self.history['val_precision'][-1]:.4f}\n")
                f.write(f"Final validation Recall: {self.history['val_recall'][-1]:.4f}\n")
                f.write(f"Final validation loss: {self.history['val_loss'][-1]:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'AdamW',
    optimizer_params: dict = None
) -> Optimizer:
    """
    Create an optimizer with adaptive configuration.
    
    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('Adam', 'AdamW', 'SGD')
        optimizer_params: Dictionary with 'lr', 'weight_decay', 'momentum' (for SGD)
        
    Returns:
        PyTorch optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(
            params,
            lr=optimizer_params.get('lr', 1e-3),
            weight_decay=optimizer_params.get('weight_decay', 1e-4)
        )
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=optimizer_params.get('lr', 1e-4),
            weight_decay=optimizer_params.get('weight_decay', 1e-4)
        )
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(
            params,
            lr=optimizer_params.get('lr', 1e-3),
            momentum=optimizer_params.get('momentum', 0.9),
            weight_decay=optimizer_params.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str = 'ReduceLROnPlateau',
    scheduler_params: dict = None,
    num_epochs: int = 50,
    steps_per_epoch: int = 100
) -> Optional[_LRScheduler]:
    """
    Create a learning rate scheduler with adaptive configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Scheduler name ('ReduceLROnPlateau', 'StepLR', 'OneCycleLR', 'CosineAnnealingLR', 'None')
        scheduler_params: Dictionary with scheduler-specific parameters
        num_epochs: Total number of epochs (for OneCycleLR, CosineAnnealingLR)
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        
    Returns:
        PyTorch scheduler or None if scheduler_name is 'None'
    """
    if scheduler_params is None:
        scheduler_params = {}
    
    # Handle 'None' scheduler (no scheduler)
    if scheduler_name is None or scheduler_name == 'None':
        return None
    
    if scheduler_name.lower() == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get('mode', 'min'),
            factor=scheduler_params.get('factor', 0.5),
            patience=scheduler_params.get('patience', 5),
            min_lr=scheduler_params.get('min_lr', 1e-7),
            verbose=False
        )
    
    elif scheduler_name.lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    
    elif scheduler_name.lower() == 'onecyclelr':
        # OneCycleLR requires total_steps calculation
        total_steps = num_epochs * steps_per_epoch
        
        # Get max_lr from optimizer
        max_lr = optimizer.param_groups[0]['lr']
        
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=scheduler_params.get('pct_start', 0.3),
            div_factor=scheduler_params.get('div_factor', 25.0),
            final_div_factor=scheduler_params.get('final_div_factor', 10000.0),
            anneal_strategy=scheduler_params.get('anneal_strategy', 'cos')
        )
    
    elif scheduler_name.lower() == 'cosineannealinglr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', num_epochs),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")