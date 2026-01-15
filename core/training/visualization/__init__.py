"""
Visualization module for QModel Trainer

Provides visualization functions for training analysis:
- Training curves (loss, metrics)
- Confusion matrix heatmaps
- LR Finder plots
- Qualitative examples generation
"""

from .plots import (
    plot_loss_curves,
    plot_metrics_curves,
    plot_confusion_matrix,
    plot_lr_finder
)
from .qualitative import (
    generate_qualitative_examples,
    colorize_mask,
    denormalize_image
)

__all__ = [
    'plot_loss_curves',
    'plot_metrics_curves',
    'plot_confusion_matrix',
    'plot_lr_finder',
    'generate_qualitative_examples',
    'colorize_mask',
    'denormalize_image'
]