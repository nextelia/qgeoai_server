"""
Plotting utilities for training visualization

Provides functions to generate professional charts:
- Loss curves (train/val)
- Metrics curves (IoU, F1, Precision, Recall)
- Confusion matrix heatmaps
- LR Finder plots
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Dict


def plot_lr_finder(
    learning_rates: List[float],
    losses: List[float],
    suggested_lr: float,
    output_path: str
) -> str:
    """
    Generate LR Finder plot (Loss vs LR).
    
    Args:
        learning_rates: List of tested learning rates
        losses: Corresponding loss values
        suggested_lr: Automatically selected LR (marked on plot)
        output_path: Full path where to save the plot
        
    Returns:
        Path to saved PNG file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot loss curve
    ax.plot(learning_rates, losses, linewidth=2, color='#2E86AB', label='Loss')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('LR Finder - Loss vs Learning Rate', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark suggested LR with vertical line
    ax.axvline(
        suggested_lr, 
        color='#A23B72', 
        linestyle='--', 
        linewidth=2.5,
        label=f'Suggested LR: {suggested_lr:.2e}'
    )
    
    # Add legend
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    output_path: str
) -> str:
    """
    Generate loss curves plot (Train vs Val).
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch (optional)
        output_path: Full path where to save the plot
        
    Returns:
        Path to saved PNG file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot train loss
    ax.plot(epochs, train_losses, linewidth=2, marker='o', markersize=4,
             label='Train Loss', color='#2E86AB', alpha=0.8)
    
    # Plot val loss with outlier clipping for better visualization
    if val_losses:
        # Detect outliers (values > 100x median)
        val_losses_clipped = []
        median_val = np.median([v for v in val_losses if v < 1000])  
        
        for v in val_losses:
            if v > median_val * 100:  # Outlier
                val_losses_clipped.append(median_val * 100)  
            else:
                val_losses_clipped.append(v)
        
        ax.plot(epochs, val_losses_clipped, linewidth=2, marker='s', markersize=4,
                 label='Val Loss', color='#A23B72', alpha=0.8)
        
        # Add note if clipping occurred
        if max(val_losses) > median_val * 100:
            ax.text(0.98, 0.98, '⚠️ Val loss clipped\n(outliers removed)', 
                    transform=ax.transAxes, fontsize=9, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_metrics_curves(
    metrics: Dict[str, List[float]],
    output_path: str
) -> str:
    """
    Generate validation metrics plot (IoU, F1, Precision, Recall).
    
    Args:
        metrics: Dict of metrics per epoch (e.g., {'IoU': [...], 'F1': [...]})
        output_path: Full path where to save the plot
        
    Returns:
        Path to saved PNG file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not metrics or len(metrics) == 0:
        # Create empty placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No metrics available', 
                ha='center', va='center', fontsize=14, color='#999')
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    # Get number of epochs from first metric
    first_metric_values = list(metrics.values())[0]
    epochs = list(range(1, len(first_metric_values) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors for each metric
    colors = {
        'IoU': '#F18F01',
        'F1': '#C73E1D', 
        'Precision': '#6A994E',
        'Recall': '#BC4B51'
    }
    
    for metric_name, values in metrics.items():
        color = colors.get(metric_name, '#666666')
        ax.plot(epochs, values, linewidth=2, marker='o', markersize=4,
                 label=metric_name, color=color, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])  # Metrics are between 0 and 1
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = True,
    cmap: str = "YlOrRd"
) -> str:
    """
    Generate confusion matrix heatmap for semantic segmentation.
    
    Pixel-based confusion matrix showing classification performance per class.
    Rows = true labels, Columns = predicted labels.
    
    Args:
        cm: Confusion matrix (C x C numpy array)
        class_names: List of class names in order
        output_path: Full path where to save the plot
        normalize: If True, normalize by row (percentage per true class)
        cmap: Matplotlib colormap name
        
    Returns:
        Path to saved PNG file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Defensive copy to avoid modifying input
    cm = cm.astype(float)
    
    # Normalize safely (avoid division by zero)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        cm = cm / row_sums
        fmt = ".2%"
        title = "Normalized Confusion Matrix (Pixel-wise)"
    else:
        fmt = "d"
        title = "Confusion Matrix (Pixel Count)"
    
    # Create figure with proper sizing
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Titles and labels
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
    
    # Add grid lines for readability
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add text annotations
    max_val = cm.max()
    threshold = max_val * 0.5
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            text = f"{value:{fmt}}" if normalize else f"{int(value)}"
            
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
                fontweight="bold"
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    
    return output_path