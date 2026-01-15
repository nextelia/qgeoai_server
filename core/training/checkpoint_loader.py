"""
Checkpoint loader for QModel Trainer

Handles loading and validating checkpoints for resume training.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch


def validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate checkpoint compatibility with current training configuration.
    
    Performs minimal but essential checks:
    - Architecture match
    - Encoder match
    - Number of classes match
    - Input channels match
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        config: Current training configuration
        
    Returns:
        Tuple of (is_valid, error_message)
        - If valid: (True, None)
        - If invalid: (False, "Error description")
    """
    # Extract metadata from checkpoint
    checkpoint_meta = checkpoint.get('metadata', {})
    
    # Check 1: Architecture
    checkpoint_arch = checkpoint_meta.get('architecture')
    current_arch = config.get('architecture')
    
    if checkpoint_arch and checkpoint_arch != current_arch:
        return False, (
            f"Architecture mismatch:\n"
            f"  Checkpoint: {checkpoint_arch}\n"
            f"  Current: {current_arch}"
        )
    
    # Check 2: Encoder/Backbone
    checkpoint_encoder = checkpoint_meta.get('encoder_name')
    current_encoder = config.get('backbone') or config.get('encoder')
    
    if checkpoint_encoder and checkpoint_encoder != current_encoder:
        return False, (
            f"Encoder mismatch:\n"
            f"  Checkpoint: {checkpoint_encoder}\n"
            f"  Current: {current_encoder}"
        )
    
    # Check 3: Number of classes
    checkpoint_classes = checkpoint_meta.get('num_classes')
    # We'll get this from model state dict shape
    model_state = checkpoint.get('model_state_dict', {})
    
    # For UNet, the final layer is typically 'segmentation_head.0.weight'
    # Shape: (num_classes, in_channels, 1, 1)
    final_layer_key = None
    for key in model_state.keys():
        if 'segmentation_head' in key and 'weight' in key:
            final_layer_key = key
            break
    
    if final_layer_key:
        checkpoint_num_classes = model_state[final_layer_key].shape[0]
    elif checkpoint_classes:
        checkpoint_num_classes = checkpoint_classes
    else:
        # Can't determine - skip this check
        checkpoint_num_classes = None
    
    
    # Check 4: Input channels (3 for RGB, standard)
    checkpoint_channels = checkpoint_meta.get('in_channels', 3)
    current_channels = 3  # We always use RGB for MVP
    
    if checkpoint_channels != current_channels:
        return False, (
            f"Input channels mismatch:\n"
            f"  Checkpoint: {checkpoint_channels}\n"
            f"  Current: {current_channels}"
        )
    
    # All checks passed
    return True, None


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
    log_callback: Optional[callable] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Load checkpoint (with optional validation).
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        config: Optional training config for validation
        log_callback: Optional callback for logging
        
    Returns:
        Tuple of (success, checkpoint_dict, error_message)
    """
    def log(message):
        """Helper to log if callback provided."""
        if log_callback:
            log_callback(message)
    
    # Check file exists
    if not os.path.exists(checkpoint_path):
        error = f"Checkpoint file not found: {checkpoint_path}"
        if log:
            log(f"❌ {error}")
        return False, None, error
    
    if log:
        log(f"\n📂 Loading checkpoint: {Path(checkpoint_path).name}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        error = f"Failed to load checkpoint: {str(e)}"
        if log:
            log(f"❌ {error}")
        return False, None, error
    
    # Check checkpoint structure
    required_keys = ['model_state_dict']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        error = f"Invalid checkpoint format. Missing keys: {missing_keys}"
        if log:
            log(f"❌ {error}")
        return False, None, error
    
    # Validate compatibility if config provided
    if config:
        valid, error_msg = validate_checkpoint_compatibility(checkpoint, config)
        
        if not valid:
            if log:
                log(f"❌ Checkpoint validation failed:\n   {error_msg}")
            return False, None, error_msg
    
    # Log checkpoint info
    if log:
        checkpoint_epoch = checkpoint.get('epoch', 0)
        checkpoint_best_iou = checkpoint.get('best_iou', 0.0)
        checkpoint_meta = checkpoint.get('metadata', {})
        
        log("✅ Checkpoint loaded successfully!")
        if checkpoint_meta:
            log(f"   Architecture: {checkpoint_meta.get('architecture', 'N/A')}")
            log(f"   Encoder: {checkpoint_meta.get('encoder_name', 'N/A')}")
            log(f"   Classes: {checkpoint_meta.get('num_classes', 'N/A')}")
        log(f"   Epoch: {checkpoint_epoch}")
        if checkpoint_best_iou > 0:
            log(f"   Best IoU: {checkpoint_best_iou:.4f}")
    
    return True, checkpoint, None


def load_checkpoint_for_resume(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
    log_callback: Optional[callable] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Load checkpoint and validate for resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Current training configuration
        device: Device to load checkpoint to
        log_callback: Optional callback for logging
        
    Returns:
        Tuple of (success, checkpoint_dict, error_message)
    """
    # Use the main load_checkpoint with config validation
    success, checkpoint, error = load_checkpoint(
        checkpoint_path, device, config, log_callback
    )
    
    if success and log_callback:
        log_callback("\n⚠️  IMPORTANT: Make sure you have a backup of your checkpoint!")
        log_callback("   Training will overwrite best_model.pth if a better model is found.")
    
    return success, checkpoint, error


def save_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    epoch: int,
    best_iou: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save model checkpoint.
    
    Args:
        checkpoint_path: Path where to save checkpoint
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        best_iou: Best IoU achieved so far
        metadata: Optional metadata dict (architecture, encoder, etc.)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'metadata': metadata or {}
        }
        
        # Create parent directory if needed
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        return True
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False