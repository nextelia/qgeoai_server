"""
Reliability Maps for QPredict

This module computes reliability metrics for semantic segmentation predictions.
It provides two complementary measures:
- Confidence: Maximum softmax probability (0=uncertain, 1=confident)
- Uncertainty: Shannon entropy of probability distribution (0=certain, high=uncertain)

These metrics help assess prediction quality and identify areas where the model
is uncertain, which is crucial for geospatial applications.
"""

import numpy as np
from scipy.special import softmax
from typing import Tuple


class ReliabilityMapCalculator:
    """
    Calculate reliability maps from model predictions.
    
    Provides two metrics:
    1. Confidence Map: max(softmax(logits)) per pixel
       - Range: [0, 1]
       - Higher = more confident
       
    2. Uncertainty Map: Shannon entropy of softmax(logits) per pixel
       - Range: [0, log(n_classes)]
       - Higher = more uncertain
    
    Args:
        num_classes: Number of classes in the model
        
    Example:
        >>> calculator = ReliabilityMapCalculator(num_classes=3)
        >>> confidence = calculator.compute_confidence(logits)
        >>> uncertainty = calculator.compute_uncertainty(logits)
    """
    
    def __init__(self, num_classes: int):
        """
        Initialize calculator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        # Pre-compute max entropy for normalization (optional)
        self.max_entropy = np.log(num_classes)
    
    def compute_confidence(self, logits: np.ndarray) -> np.ndarray:
        """
        Compute confidence map (max softmax probability).
        
        This represents "How confident is the model in its prediction?"
        Higher values indicate higher confidence.
        
        Args:
            logits: Raw model outputs of shape (C, H, W)
        
        Returns:
            Confidence map of shape (H, W) with values in [0, 1]
        """
        # Apply softmax along class dimension
        probs = softmax(logits, axis=0)  # (C, H, W)
        
        # Take maximum probability per pixel
        confidence = np.max(probs, axis=0)  # (H, W)
        
        return confidence.astype(np.float32)
    
    def compute_uncertainty(self, logits: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        Compute uncertainty map (Shannon entropy).
        
        This represents "How uncertain is the model's probability distribution?"
        Higher values indicate more uncertainty (flatter distribution).
        
        Shannon entropy formula:
        H(p) = -sum(p_i * log(p_i)) for all classes i
        
        Args:
            logits: Raw model outputs of shape (C, H, W)
            normalized: If True, normalize entropy to [0, 1] range
        
        Returns:
            Uncertainty map of shape (H, W)
            - If normalized=False: values in [0, log(n_classes)]
            - If normalized=True: values in [0, 1]
        """
        # Apply softmax along class dimension
        probs = softmax(logits, axis=0)  # (C, H, W)
        
        # Compute Shannon entropy per pixel
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs_safe = np.clip(probs, epsilon, 1.0)
        
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=0)  # (H, W)
        
        # Optional normalization to [0, 1]
        if normalized:
            entropy = entropy / self.max_entropy
        
        return entropy.astype(np.float32)
    
    def compute_both(
        self,
        logits: np.ndarray,
        normalize_uncertainty: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both confidence and uncertainty maps.
        
        This is more efficient than calling both methods separately
        as it only computes softmax once.
        
        Args:
            logits: Raw model outputs of shape (C, H, W)
            normalize_uncertainty: Whether to normalize uncertainty to [0, 1]
        
        Returns:
            Tuple of (confidence, uncertainty) maps, both of shape (H, W)
        """
        # Apply softmax once
        probs = softmax(logits, axis=0)  # (C, H, W)
        
        # Confidence: max probability
        confidence = np.max(probs, axis=0)  # (H, W)
        
        # Uncertainty: entropy
        epsilon = 1e-10
        probs_safe = np.clip(probs, epsilon, 1.0)
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=0)  # (H, W)
        
        if normalize_uncertainty:
            entropy = entropy / self.max_entropy
        
        return confidence.astype(np.float32), entropy.astype(np.float32)